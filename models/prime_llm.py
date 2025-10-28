import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from .embeddings import LinearEmbeddingLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PRIME_LLM(nn.Module):
    """
    Adapter model that embeds tabular/time-series inputs and feeds them into a pretrained
    causal LM via inputs_embeds. Handles optional LoRA, projects embedding to model hidden_size
    when necessary, and safely uses autocast only on CUDA.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        pretrained_model: str,
        lora_target_modules,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        num_classes: int = None,
    ):
        super().__init__()

        # float32 embedding adapter
        self.embedding = LinearEmbeddingLayer(input_dim, hidden_dim, embedding_dim, dropout)
        self.embed_norm = nn.LayerNorm(embedding_dim)

        # load base transformer (may be placed on GPU automatically if available)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # freeze base parameters by default (LoRA will add trainable params)
        for p in self.transformer.parameters():
            p.requires_grad = False

        # attach LoRA adapters if requested
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.transformer = get_peft_model(self.transformer, lora_cfg)
            # get_peft_model will make some parameters trainable; printing optional
            try:
                self.transformer.print_trainable_parameters()
            except Exception:
                pass

        self.dropout = nn.Dropout(dropout)

        # ensure head dimension matches transformer hidden size
        model_hidden_size = self.transformer.config.hidden_size
        if embedding_dim != model_hidden_size:
            self.input_proj = nn.Linear(embedding_dim, model_hidden_size)
            nn.init.kaiming_normal_(self.input_proj.weight)
        else:
            self.input_proj = None

        # unified head
        if num_classes is None:
            self.head = nn.Linear(model_hidden_size, 1)
        else:
            self.head = nn.Linear(model_hidden_size, num_classes)

        # move lightweight modules to CPU/GPU consistent with transformer device
        try:
            transformer_device = next(self.transformer.parameters()).device
        except StopIteration:
            transformer_device = device
        self.to(transformer_device)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor):
        """
        X: (batch, seq_len, input_dim)
        attention_mask: (batch, seq_len) with 1 for valid positions, 0 for padding
        """
        # ensure tensors are on same device as transformer
        target_device = next(self.transformer.parameters()).device
        X = X.to(target_device)
        attention_mask = attention_mask.to(target_device)

        # embedding in float32 for numerical stability
        X_embeds = self.embedding(X.float(), attention_mask.float())
        X_embeds = self.embed_norm(X_embeds)

        # project to transformer's hidden size if necessary
        if self.input_proj is not None:
            X_embeds = self.input_proj(X_embeds)

        # cast embeds to transformer's dtype (fp16 if GPU, fp32 otherwise)
        X_embeds = X_embeds.to(next(self.transformer.parameters()).dtype)

        # attention_mask expected by transformer can be bool/long
        attn_mask = attention_mask.bool()

        # Forward through transformer with autocast only when CUDA is available
        if torch.cuda.is_available():
            with amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.transformer(inputs_embeds=X_embeds, attention_mask=attn_mask, output_hidden_states=True)
        else:
            outputs = self.transformer(inputs_embeds=X_embeds, attention_mask=attn_mask, output_hidden_states=True)

        # obtain pooled representation; causal LM may not provide pooler_output
        pooled_output = None
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # use last hidden state and take last token embedding
            hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
            pooled_output = hidden[:, -1, :]

        logits = self.head(self.dropout(pooled_output))
        return logits