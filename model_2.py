import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_keys = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        key = self.linear_keys(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_values(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        query = self.linear_query(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        attention = self.softmax(scores)
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        return context

class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, num_heads=6):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.affect_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256 
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_affect = nn.Embedding(self.emb_num, self.affect_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.attention = MultiHeadAttention(num_heads, self.knowledge_dim * 2)
        self.prednet_affect = nn.Linear(self.knowledge_dim + self.affect_dim, 4)
        self.guess = nn.Linear(4, 1)
        self.slip = nn.Linear(4, 1)
        self.prednet_full1 = nn.Linear(408, 512)  # Adjusted to fit the output from previous layer
        self.prednet_full2 = nn.Linear(512, 256)  # Now correctly expecting 512 inputs
        self.prednet_full3 = nn.Linear(256, 1)
        self.kn_emb_transform = nn.Linear(102, 512)
        self.dropout = nn.Dropout(p=0.25)
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        combined_emb = torch.cat((stu_emb, k_difficulty), dim=1)
        attention_output = self.attention(combined_emb, combined_emb, combined_emb)
        attention_output = nn.LayerNorm(attention_output.size())(attention_output)
        attention_output = attention_output.view(attention_output.size(0), -1)
        combined_features = torch.cat((attention_output, kn_emb), dim=1)
        refined_input = torch.cat([combined_features, stu_emb - k_difficulty], dim=1)
        refined_input = torch.sigmoid(self.prednet_full1(refined_input))
        kn_emb = self.kn_emb_transform(kn_emb)
        input_x = e_discrimination * refined_input * kn_emb
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))
        affect = torch.sigmoid(self.prednet_affect(torch.cat((stu_affect, k_difficulty), dim=1)))
        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        output = ((1-s)*o) + (g*(1-o))
        return output, affect

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(min=0)
