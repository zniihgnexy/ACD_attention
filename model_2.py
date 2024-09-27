import torch
import torch.nn as nn
import math

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
        
        # Student and exercise embeddings
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_affect = nn.Embedding(self.emb_num, self.affect_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        
        # Fully connected layers for adjusting dimension mismatches
        self.fc_adjust_student = nn.Linear(self.stu_dim, 256)  # Adjust student embedding to 256
        self.fc_adjust_knowledge = nn.Linear(self.knowledge_dim, 256)  # Adjust knowledge embedding to 256
        self.fc_adjust_affect = nn.Linear(self.affect_dim, 256)  # Adjust affect embedding to match 256
        self.fc_adjust_combined = nn.Linear(1024, 512)  # New fully connected layer to adjust input size
        
        # Prediction and comparison network layers
        self.prednet_affect = nn.Linear(512, 4)  # Adjust to combine both embeddings after they are of equal size
        self.guess = nn.Linear(4, 1)
        self.slip = nn.Linear(4, 1)
        
        # Main comparison network
        self.prednet_full1 = nn.Linear(512, 512)  # Adjusted input size after transformation
        self.prednet_full2 = nn.Linear(512, 256)
        self.prednet_full3 = nn.Linear(256, 1)
        self.kn_emb_transform = nn.Linear(102, 512)
        
        self.dropout = nn.Dropout(p=0.25)
        
        # Initialize weights using Xavier initialization
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        # Get student and exercise embeddings
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        stu_emb = self.fc_adjust_student(stu_emb)  # Adjust student embedding size
        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        stu_affect = self.fc_adjust_affect(stu_affect)  # Adjust affect embedding size
        
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        k_difficulty = self.fc_adjust_knowledge(k_difficulty)  # Adjust knowledge embedding size
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        
        # Compare student and exercise by concatenation
        combined_features = torch.cat((stu_emb, k_difficulty), dim=1)
        
        # Apply transformation to knowledge embedding
        kn_emb = self.kn_emb_transform(kn_emb)
        
        # Refined input combines all features
        refined_input = torch.cat([combined_features, kn_emb], dim=1)
        refined_input = self.fc_adjust_combined(refined_input)  # Adjust size before feeding to the next layer
        
        # Pass through the comparison network layers
        refined_input = torch.sigmoid(self.prednet_full1(refined_input))
        refined_input = torch.sigmoid(self.prednet_full2(refined_input))
        o = torch.sigmoid(self.prednet_full3(refined_input))
        
        # Affect prediction remains the same
        affect = torch.sigmoid(self.prednet_affect(torch.cat((stu_affect, k_difficulty), dim=1)))
        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        
        # Combine slip, guess, and o to get the final output
        output = ((1 - s) * o) + (g * (1 - o))
        
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
