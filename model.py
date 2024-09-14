import math
from numpy import size
import torch
import torch.nn as nn

# logs
# 0.25 with one dropout, 0.7253
# 0.25 with two dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        super(MultiHeadAttention, self).__init__()
        # breakpoint()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_keys = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, num_heads * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        # Transform inputs to multi-head format
        key = self.linear_keys(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_values(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        query = self.linear_query(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate and apply attention
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
        self.prednet_len1, self.prednet_len2 = 512,256 
        
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_affect = nn.Embedding(self.emb_num, self.affect_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        
        self.attention = MultiHeadAttention(num_heads, self.knowledge_dim * 2)  # *2 as we concat student and exercise embeddings
        
        self.prednet_affect = nn.Linear(self.knowledge_dim + self.affect_dim, 4)
        self.guess = nn.Linear(4, 1)
        self.slip = nn.Linear(4, 1)
        
        # Adjust the input feature size here to match the output from attention and the dimension of kn_emb
        # self.prednet_full1 = nn.Linear(self.knowledge_dim * 2 * num_heads, 512)
        self.prednet_full1 = nn.Linear(102, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
        self.prednet_full4 = nn.Linear(358, 102)
        self.dropout = nn.Dropout(p=0.25)  # p 是dropout概率
        self.layer_norm1 = nn.LayerNorm(self.prednet_len1)
        self.layer_norm2 = nn.LayerNorm(self.prednet_len2)
        self.bn1 = nn.BatchNorm1d(self.prednet_len1)
        self.bn2 = nn.BatchNorm1d(self.prednet_len2)
        # p = 0.6 not so good, acc around0.722 and stopped, p=0.45 is higher at 0.723
        # p=0.25 is the best, acc around 0.7253
        # p = 0.2 acc around 0,7238 p = 0.15 acc around 0.732
        # p=0.3
        
        self.emotion_extractor = nn.Sequential(
            nn.Linear(self.affect_dim, 256),
            # nn.ReLU(),
            nn.Linear(256, 64),
            # nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.emotion_combiner = nn.Linear(268, 256)
        # self.prednet_full1 = nn.Linear(256 + self.knowledge_dim, self.prednet_len1)


        # Weight initialization
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)
        # Weight initialization
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) == 2:  # Ensure the parameter is 2D
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, exer_id, kn_emb):
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        
        combined_emb = torch.cat((stu_emb, k_difficulty), dim=1)
        attention_output = self.attention(combined_emb, combined_emb, combined_emb)
        attention_output = attention_output.view(attention_output.size(0), -1)
        
        emotional_features = self.emotion_extractor(stu_affect)
        emotional_context = torch.cat([attention_output, emotional_features], dim=1)
        emotional_context = self.emotion_combiner(emotional_context)

        combined_features = torch.cat((emotional_context, kn_emb), dim=1)
        
        # Ensure the input dimensions match the expectation for the next layer
        refined_input = self.prednet_full4(combined_features)  # Updated layer with correct dimensions
        refined_input = torch.sigmoid(refined_input)  # Activation function

        input_x = e_discrimination * refined_input * kn_emb
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = self.dropout(input_x)
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

class NoneNegClipper(object):  # Ensures non-negative weights
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(min=0)