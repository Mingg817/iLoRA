import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from .peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, MoeLoraConfig, MoeLoraModel
import pickle
from .router.nlpr import LambdaLayer, ResidualBlock, GateFunction, NLPRecommendationRouter, build_router


# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
class MInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        # 保存模型超参数
        self.save_hyperparameters()
        # 加载LLM
        self.load_llm(self.hparams.llm_path)
        
        # 加载Router
        if self.hparams.router == 'share':
            self.router = build_router()
        
        # 加载推荐模型
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()
        self.gradient_storage = {}
    
    # 单次前向传播forward
    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,:], -100)
        
        input_embeds, user_embeds = self.wrap_emb(batch)
        # print("user_embeds before in batch:", user_embeds)

        # user_embeds_rand = torch.randn_like(user_embeds).to(input_embeds.device)
        if self.hparams.router == 'share':
            # print("user_embeds.shape: ", user_embeds.shape)
            gate_weights = self.router(user_embeds)
            # print("gate_weights.shape: ", gate_weights.shape)
            # print("weights: ", gate_weights)
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=batch["tokens"].attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,
                user_embeds=user_embeds,
                gate_weights=gate_weights
            )
            return outputs
        
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
            user_embeds=user_embeds
        )
        return outputs

    # 根据batch, LLM生成输出outputs
    def generate(self, batch,temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        input_embeds, user_embeds = self.wrap_emb(batch)
        # print("input_embeds.shape: ",input_embeds.shape)
        # print("user_embeds.shape: ", user_embeds.shape)
        """if self.hparams.if_rand == True:
            print("rand")
            user_embeds = torch.randn_like(user_embeds).to(input_embeds.device)
            print("rand user_embeds:", user_embeds)"""
            
        # user_embeds_rand = torch.randn_like(user_embeds).to(input_embeds.device)
        # print("user_embeds in batch:", user_embeds)
        if self.hparams.router == 'share':
            # print("user_embeds.shape: ", user_embeds.shape)
            gate_weights = self.router(user_embeds)
            # print("gate_weights.shape: ", gate_weights.shape)
            # print("weights: ", gate_weights)
            generate_ids = self.llama_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=batch["tokens"].attention_mask,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                max_new_tokens=max_gen_length,
                min_new_tokens=min_gen_length,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                user_embeds=user_embeds,
                gate_weights = gate_weights
            )
            output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs=[text.strip() for text in output_text]
            return outputs
            
        gate_weights = self.router(user_embeds)
        
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            
            user_embeds=user_embeds, 
            gate_weights = gate_weights
            )
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs
    
    def capture_and_store_gradients(self):
        for name, param in self.llama_model.named_parameters():
            if "lora" in name and param.grad is not None:
                if name not in self.gradient_storage:
                    self.gradient_storage[name] = []
                self.gradient_storage[name].append(param.grad.clone().detach())
        
        # 定义保存文件的条件，例如每累积一定数量的梯度后保存
        if self.trainer.global_step % 10 == 0:  # 每200步保存一次梯度数据
            self.save_gradients_to_file()
            
    def save_gradients_to_file(self):
        directory = self.hparams.capture_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'gradients_step_{self.trainer.global_step}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self.gradient_storage, f)
        self.gradient_storage = {}  # 清空存储以避免内存泄漏
            
    # 单步训练
    def training_step(self, batch, batch_idx):
        # scheduler更新学习率
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        # 若batch["flag"] -> easy task -> projector冻结
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        else:
        # 否则 -> hard task -> projector可训
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
        # 调用forward得到out
        out = self(batch)
        # 计算loss
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr_gate', self.scheduler.optimizer.param_groups[2]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
            
    # 验证轮次开始调用, 创建generate生成结果, real真实结果, cans候选结果
    def on_validation_epoch_start(self):
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # 生成模型输出
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        # 返回output, 包含generate, real, cans的元组
        return output

    # 验证批次结束调用
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # 遍历outputs, 分解为generate/real/cans三部分
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)
            # 作为字典存储

    # 验证轮次结束调用
    def on_validation_epoch_end(self):
        # val_content -> df 存储val_data(generate, real, cans)结果为valid.csv
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        # 计算pvr, hr
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        ctr = self.calculate_ctr(self.val_content)
        # 计算metic: 有效比率 * 命中率
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('ctr', ctr, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio
        
        ctr = self.calculate_ctr(self.test_content)
        
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('ctr', ctr, on_step=False, on_epoch=True, prog_bar=True)

        
    # 配置优化器&学习率调度器
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            
            # 为router设置默认的学习率和权重衰减
            {'params': self.router.parameters(), 'lr': self.hparams.lr * 0.3, 'weight_decay':weight_decay},
            
            # 为非 gating 参数设置默认的学习率和权重衰减
            {'params': [p for n, p in self.llama_model.named_parameters() if "gating" not in n], 'lr': self.hparams.lr},
            # 为所有包含 'gating' 的参数设置较低的学习率: 0.01, 0.05, 0.1
            # {'params': [p for n, p in self.llama_model.named_parameters() if "gating" in n], 'lr': self.hparams.lr * 1, 'weight_decay':weight_decay}
           
            # {'params': self.llama_model.parameters(), 'lr': self.hparams.lr},
        ])
        
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Initial LR for group {i}: {param_group['lr']}")   
            # 计算每个参数组的总参数数量
            total_params = sum(p.numel() for p in param_group['params'])
            print(f"Parameter Group {i}: {total_params} parameters")

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                
                # 定义每个参数组的学习率设置
                init_lr_list = [
                    self.hparams.lr,  # 对 projector 参数
                    self.hparams.lr * 0.3,  # 对 router 参数
                    self.hparams.lr * 1  # 对 llama_model 非 gating 参数
                ]
                min_lr_list = [
                    self.hparams.lr_decay_min_lr,  # 对 projector 参数
                    self.hparams.lr_decay_min_lr * 0.3,  # 对 router 参数
                    self.hparams.lr_decay_min_lr * 1  # 对 llama_model 非 gating 参数，如果需要更低的 min_lr
                ]
                warmup_start_lr_list = [
                    self.hparams.lr_warmup_start_lr,  # 对 projector 参数
                    self.hparams.lr_warmup_start_lr * 0.3,  # router 参数
                    self.hparams.lr_warmup_start_lr * 1  # 对 llama_model 非 gating 参数
                ]
                # 创建调度器实例
                self.scheduler = LinearWarmupCosineLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    min_lr_list=min_lr_list,
                    init_lr_list=init_lr_list,
                    warmup_steps=warmup_steps,
                    warmup_start_lr_list=warmup_start_lr_list
                )
                                
                
                print("after schedule")
                for i, param_group in enumerate(optimizer.param_groups):
                    print(f"Initial LR for group {i}: {param_group['lr']}")   
                    # 计算每个参数组的总参数数量
                    total_params = sum(p.numel() for p in param_group['params'])
                    print(f"Parameter Group {i}: {total_params} parameters")
                    
                    
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    # 计算loss
    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")
    
    # 计算user_emb -> gate_weight

    def configure_loss_moe(self, out, lables=None, gate_outpus=None):
        loss = self.hparams.loss.lower()
        # 主损失
        if loss == 'lm':
            main_loss = out.loss
        else:
            raise ValueError("Invalid Loss Type!")
        # MOE特定的loss计算
        if gate_outpus is not None:
            # 计算负载平衡损失
            load_balancing_loss = self.calculate_load_balancing_loss(gate_outpus)
            # 计算容量损失
            capacity_loss = self.calculate_capacity_loss(gate_outpus)
            # 结合主loss和MOE loss
            total_loss = main_loss + self.hparams.load_balancing_weight * load_balancing_loss + self.hparams.capacity_weight * capacity_loss
        else:
            total_loss = main_loss
        return total_loss
    
    def calculate_load_balancing_loss(self, gate_outputs):
        # 实现计算负载平衡损失
        expert_usage = gate_outputs.mean(dim=0)  # 计算每个专家的平均使用率
        # 在这里，alpha是一个超参数，用于调节额外惩罚项的权重，threshold是专家使用率的期望最小阈值。
        # 这种方法试图在减少使用率的标准差的同时，提高那些使用率低于某个阈值的专家的使用率。
        loss = torch.std(expert_usage) + self.alpha * torch.mean(torch.relu(self.threshold - expert_usage))
        return loss

    def calculate_capacity_loss(self, gate_outputs):
        # 计算容量损失
        # 这也是一个示例，具体实现需根据模型的需求调整
        # 假设有一个最大容量限制
        max_capacity = 0.1
        expert_usage = gate_outputs.mean(dim=0)
        capacity_loss = torch.relu(expert_usage - max_capacity).mean()  # 超出容量部分的平均值作为损失
        return capacity_loss
    
    # 保存检查点时调用
    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass
        
    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        # 添加特殊token: '[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]'
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        # , load_in_8bit=True, device_map="auto"
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path,torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            # 若tuning为LoRA模式
            if self.hparams.peft_dir:
                # 若peft_dir不为none, 加载检查点LoRA
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                # 否则, 初始化LoRA
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            # 若tuning为freeze模式, 不加载LoRA, 冻结LLM
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            # 若tuning为freeze_lora
            if self.hparams.peft_dir:
                # 加载检查点LoRA
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                # 初始化LoRA
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            # 冻结LoRA权重
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'moelora':
            # 若tuning为MOELoRA模式
            if self.hparams.peft_dir:
                # 若peft_dir不为none, 加载检查点LoRA
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                # 否则, 初始化MOELoRA
                if self.hparams.peft_config:
                    peft_config = MoeLoraConfig(**MoeLoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = MoeLoraConfig(task_type=TaskType.CAUSAL_LM,
                                                inference_mode=False,
                                                r=self.hparams.lora_r,
                                                lora_alpha=self.hparams.lora_alpha,
                                                lora_dropout=self.hparams.lora_dropout,
                                                target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                                                num_moe=self.hparams.num_moe,
                                                gating=self.hparams.gating)
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                
                """for name, param in self.llama_model.named_parameters():
                    if "gating" not in name:
                        param.requires_grad = False"""
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()
 
        print('Loading LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        
        # 
        print("camel_name = ", camel_name)
        
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        # 实例化模型
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    # 实例化模型方法
    def instancialize(self, Model, **other_args):
        # 获取模型初始化方法的参数列表class_args
        class_args = inspect.getargspec(Model.__init__).args[1:]
        # inkeys: hparams所有键
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        # args1: args在hparams中有的部分
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        # torch.load加载模型
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        # 冻结参数
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    # Seq模型编码item embedding
    def encode_items(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        # 通过projector得到item_txt_embs
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
    
    # Seq模型编码user embedding
    def encode_users(self, seq, len_seq):
        if self.hparams.rec_embed=="SASRec":
            user_rec_embs=self.rec_model.cacul_h(seq, len_seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            
            # 待修改
            user_rec_embs=self.rec_model.item_embeddings(seq)
        
        user_txt_embs=self.projector(user_rec_embs)    
        return user_rec_embs
    
    # LLM模型编码token embedding
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    # batch -> embeds
    def wrap_emb(self, batch):
        # token_ids -> input_embeds嵌入向量
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        
           
        
        # llama_tokenizer编码特殊token: [HistoryEmb],[CansEmb],[ItemEmb],[UserEmb]
        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
         
        # 额外编码特殊token: [UserEmb] 
        
        # user_token_id=self.llama_tokenizer("[UserEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        
        # Seq模型通过encode_items得到his_item_embeds, cans_item_embeds, item_embeds
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
        
        
        # 额外获取user_embeds
        user_embeds=self.encode_users(batch["seq"], batch["len_seq"])
        # 随机user_embeds
        # shape = user_embeds.shape
        # user_embeds = torch.rand(shape, dtype=user_embeds.dtype, device=user_embeds.device)
        """target_dim = self.llama_model.config.hidden_size
        repeat_time = target_dim // user_embeds.size(-1)
        user_embeds = user_embeds.repeat(1, 1, repeat_time)"""
        
        # 遍历batch["len_seq"], 若token_id为特殊token, 替换为对应item_embeds    
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        
            # 额外处理user_embeds    
            """if (batch["tokens"].input_ids[i]==user_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==user_token_id).nonzero().item()
                input_embeds[i,idx]=user_embeds[i]
                print("idx = ",idx)
                print("input_embeds[i,idx] = ",input_embeds[i,idx])
                print("user_embeds[i] = ",user_embeds[i])"""
            
        return input_embeds, user_embeds
     
    # 计算valid_ratio和hr1
    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        for i,generate in enumerate(eval_content["generate"]):
            real=eval_content["real"][i]
            cans=eval_content["cans"][i]
            total_num+=1
            generate=generate.strip().lower().strip()
            real=real.strip().lower().strip()
            cans=[item.strip().lower().strip() for item in cans]
            gen_cans_list=[]
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list)==1:
                valid_num+=1
                if real == gen_cans_list[0]:
                    correct_num+=1
        valid_ratio=valid_num/total_num
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio,hr1
    
    def calculate_ctr(self, eval_content):
        correct_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content['generate']):
            real = eval_content["real"][i]
            total_num += 1
            generate=generate.strip().lower().strip()
            real=real.strip().lower().strip()
            if generate == real:
                correct_num += 1
        ctr = correct_num/total_num
        return ctr