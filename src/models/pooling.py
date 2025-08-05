import torch
from torch import nn

class AttentionPoolPerImage(nn.Module):
    def __init__(self, input_dim, out='mean', pool_layer ='linear'):
        super(AttentionPoolPerImage, self).__init__()
        self.pool_layer = pool_layer
        if self.pool_layer =='linear':
            self.linear = nn.Linear(input_dim, 1, bias=False)
        else: 
            self.attention_weights = nn.Parameter(torch.randn(input_dim))  # Trainable attention weights
        self.out = out
    def forward(self, embeddings):
        
        # Calculate attention scores using dot product with trainable weights
        attention_scores = self.linear(embeddings).squeeze(-1) if self.pool_layer =='linear' else torch.matmul(embeddings, self.attention_weights)
        
        # Normalize attention scores using softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to embeddings
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)
        
        if self.out=='max':
            output = torch.max(weighted_embeddings, dim=1).values
        elif self.out=='sum':
            output = torch.sum(weighted_embeddings, dim=1)
        else:
            output = torch.mean(weighted_embeddings, dim=1)
    
        return output
    
class AttentionPoolPerDimension(nn.Module):
    def __init__(self, input_dim, out='mean', pool_layer ='linear'):
        super(AttentionPoolPerDimension, self).__init__()
        self.pool_layer = pool_layer
        if self.pool_layer =='linear':
            self.linear = nn.Linear(input_dim, input_dim, bias=False)
        else: 
            self.attention_weights = nn.Parameter(torch.randn(input_dim,input_dim))  # Trainable attention weights for each dimension
        self.out = out

    def forward(self, embeddings):
      
        # Calculate attention scores using dot product with trainable weights 
        attention_scores = self.linear(embeddings) if self.pool_layer =='linear' else torch.matmul(embeddings, self.attention_weights)
        
        # Normalize attention scores using softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
       
       
        # Apply attention weights to embeddings
        weighted_embeddings = embeddings * attention_weights
       
        # Calculate the final output based on the specified aggregation function
        if self.out == 'max':
            output = torch.max(weighted_embeddings, dim=1).values
        elif self.out == 'sum':
            output = torch.sum(weighted_embeddings, dim=1)
        else:
            output = torch.mean(weighted_embeddings, dim=1)

        return output