import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from transformers import BertModel, BertConfig, AlbertModel, AlbertConfig, XLNetConfig, XLNetModel, ElectraConfig, ElectraModel
import os
import torch.nn.functional as F
from diff_transformer import MultiheadDiffAttn, MultiheadSelfDiffAttn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)

    def load_best(self):
        print(f"loading best checkpoint from file {self.save_dir}/best.pt")
        state_dict = torch.load(os.path.join(self.save_dir, 'best.pt'))
        self.load_state_dict(state_dict, strict=False)


class MMModel(BaseModel):
    def __init__(self, imageEncoder, textEncoder, save_dir):
        super(MMModel, self).__init__(save_dir=save_dir)
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x):
        raise NotImplemented


class TextOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_text_repr=768, num_class=2):
        super(TextOnlyModel, self).__init__(save_dir)
        
        self.dropout = nn.Dropout()
        #config = BertConfig()
        #self.textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        #config = AlbertConfig()
        #self.textEncoder = AlbertModel(config).from_pretrained('albert-base-v2')
        config = ElectraConfig()
        self.textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        #config = XLNetConfig()
        #self.textEncoder = XLNetModel(config).from_pretrained('xlnet-base-cased')



        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x):
        _, text = x

        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr

        return self.linear(e_i)


class ImageOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_visual_repr=1000, num_class=2):
        super(ImageOnlyModel, self).__init__(save_dir=save_dir)

        self.imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        image, _ = x

        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        return self.linear(f_i)


class DenseNetBertMMModel122(MMModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=1000, dim_text_repr=768, dim_proj=128, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr


       
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
       
        #config = BertConfig()
        #textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        #config = AlbertConfig()
        #textEncoder = AlbertModel(config).from_pretrained('albert-base-v2')
        config = ElectraConfig()
        textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        #config = XLNetConfig()
        #textEncoder = XLNetModel(config).from_pretrained('xlnet-base-cased')


        super(DenseNetBertMMModel122, self).__init__(imageEncoder, textEncoder, save_dir)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        
        
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)


        # self.multihead_attn1 = nn.MultiheadAttention(dim_proj, num_heads=4, batch_first=True)
        # self.multihead_attn2 = nn.MultiheadAttention(dim_proj, num_heads=4, batch_first=True)
        
        # self.fc_as_self_attn = nn.Linear(dim_proj, dim_proj)
        # self.self_attn_bn = nn.BatchNorm1d(dim_proj)
        # self.cls_layer = nn.Linear(dim_proj, num_class)
        
        
        self.multihead_diffattn = MultiheadDiffAttn(embed_dim=2*dim_proj, depth=1, num_heads=4)

        self.multihead_diffattn_img = MultiheadDiffAttn(embed_dim=dim_visual_repr, depth=1, num_heads=4)
        self.multihead_diffattn_txt = MultiheadDiffAttn(embed_dim=dim_text_repr, depth=1, num_heads=4)

        


    def build_rel_pos(self, x, start_pos = 0):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / ((10000 * self.posi_scale) ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.flatten_input_size).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))


    def forward(self, x):
        image, text = x


        # print("image shape", image.shape)
        # print("text shape", text.shape)

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))



        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr

        
        # Applying self-attention to f_i and e_i
        # f_i_self_attn = self.apply_self_attention(f_i)  # N, dim_visual_repr
        # e_i_self_attn = self.apply_self_attention(e_i)  # N, dim_text_repr

        # print('f_i_self_attn', f_i_self_attn.shape)
        # print('e_i_self_attn', e_i_self_attn.shape)




        # print('img_embedding', img_embedding.shape)
        # print('text_embedding', text_embedding.shape)
        # img_embedding torch.Size([8, 784, 1152])
        # text_embedding torch.Size([8, 512, 768])
                
        f_i_self_attn = self.multihead_diffattn_img(f_i, f_i)  # N, 
        e_i_self_attn = self.multihead_diffattn_txt(e_i, e_i)  # N, 


        
        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i)))  # N, dim_proj


        # f_i_self_attn = self.multihead_diffattn_img(f_i_tilde, f_i_tilde)  # N, 
        # e_i_self_attn = self.multihead_diffattn_txt(e_i_tilde, e_i_tilde)  # N, 



        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        
        # ####################
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)  # N, 2*dim_proj
        ##########################

        # cros_attn1, _ = self.multihead_attn1(query=alpha_v_i, key=f_i_tilde, value=f_i_tilde, need_weights=False)
        # cros_attn2, _ = self.multihead_attn2(query=alpha_e_i, key=e_i_tilde, value=e_i_tilde, need_weights=False)
        

        # print("joint_repr", joint_repr.shape)
        # joint_repr torch.Size([8, 1296, 64])

        cros_attn1 = self.multihead_diffattn(joint_repr, joint_repr)

        # print("cros_attn1", cros_attn1.shape)
        # cros_attn1 torch.Size([8, 1296, 8])

        # x = self.mlp_head(cros_attn1[:, 0, :])

        # joint_repr = torch.cat((cros_attn1, cros_attn2),dim=1)  # N, 2*dim_proj
        # Get class label prediction logits with final fully-connected layers

        # return x
        return self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(cros_attn1 )))))




class AttentionOnEmbedding(nn.Module):
    def __init__(self, embed_dim):
        """
        Initializes the attention mechanism for feature embeddings.
        Args:
            embed_dim: Dimensionality of the input embedding vector.
        """
        super(AttentionOnEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Learnable weights for Query, Key, Value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the attention mechanism.
        Args:
            x: Input tensor of shape (batch_size, embed_dim).
        Returns:
            Output tensor of shape (batch_size, embed_dim).
        """
        # Compute Query, Key, Value projections
        Q = self.query_proj(x)  # (batch_size, embed_dim)
        K = self.key_proj(x)    # (batch_size, embed_dim)
        V = self.value_proj(x)  # (batch_size, embed_dim)

        # Compute attention scores (scaled dot-product)
        attention_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))  # (batch_size, 1, 1)
        attention_scores = attention_scores / (self.embed_dim ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, 1)

        # Compute the weighted sum of values
        attended_values = attention_weights.squeeze(-1) * V  # (batch_size, embed_dim)

        return attended_values



class CrossAttentionSingleQuery(nn.Module):
    def __init__(self, embed_dim):
        """
        Initializes the cross-attention mechanism for single-query attention.
        Args:
            embed_dim: Dimensionality of the embeddings (shared across query, key, value).
        """
        super(CrossAttentionSingleQuery, self).__init__()
        self.embed_dim = embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)  # Projection for query
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # Projection for keys
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # Projection for values

    def forward(self, query, keys, values):
        """
        Forward pass for the cross-attention mechanism.
        Args:
            query: Query tensor of shape (batch_size, embed_dim).
            keys: Keys tensor of shape (batch_size, embed_dim).
            values: Values tensor of shape (batch_size, embed_dim).
        Returns:
            Output tensor of shape (batch_size, embed_dim).
        """
        # Project query, keys, and values
        Q = self.query_proj(query)  # (batch_size, embed_dim)
        K = self.key_proj(keys)    # (batch_size, embed_dim)
        V = self.value_proj(values)  # (batch_size, embed_dim)

        # Compute attention scores (batch_size, 1)
        attention_scores = torch.sum(Q * K, dim=-1, keepdim=True) / (self.embed_dim ** 0.5)  # (batch_size, 1)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1)

        # Compute the weighted sum of values
        output = attention_weights * V  # (batch_size, embed_dim)

        return output


        
class DenseNetBertMMModel(MMModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=1000, dim_text_repr=768, dim_proj=128, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr


       
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
       
        #config = BertConfig()
        #textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        #config = AlbertConfig()
        #textEncoder = AlbertModel(config).from_pretrained('albert-base-v2')
        config = ElectraConfig()
        textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        #config = XLNetConfig()
        #textEncoder = XLNetModel(config).from_pretrained('xlnet-base-cased')


        super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder, save_dir)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        
        
        # self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        # self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # # Classification layer
        # self.cls_layer = nn.Linear(2*dim_proj, num_class)


        # self.multihead_attn1 = nn.MultiheadAttention(2*dim_proj, num_heads=4, batch_first=True)
        # self.multihead_attn2 = nn.MultiheadAttention(dim_proj, num_heads=4, batch_first=True)
        # self.self_attn_last = AttentionOnEmbedding(embed_dim=2*dim_proj)
        # self.self_attn_last = MultiheadDiffAttn(embed_dim=2*dim_proj, depth=1, num_heads=4)
        self.cross_attn_last = MultiheadDiffAttn(embed_dim=dim_proj, depth=1, num_heads=4)
        # self.self_attn_last = CrossAttentionSingleQuery(embed_dim=dim_proj)

        self.fc_as_self_attn = nn.Linear(dim_proj, dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(dim_proj)
        self.cls_layer = nn.Linear(dim_proj, num_class)
        
        




    def forward(self, x):
        image, text = x


        # print("image shape", image.shape)
        # print("text shape", text.shape)

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))



        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr

        
        # Applying self-attention to f_i and e_i
        f_i_self_attn = self.apply_self_attention(f_i)  # N, dim_visual_repr
        e_i_self_attn = self.apply_self_attention(e_i)  # N, dim_text_repr

        # print('f_i_self_attn', f_i_self_attn.shape)
        # print('e_i_self_attn', e_i_self_attn.shape)




        # print('img_embedding', img_embedding.shape)
        # print('text_embedding', text_embedding.shape)
        # img_embedding torch.Size([8, 784, 1152])
        # text_embedding torch.Size([8, 512, 768])
                

        
        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i_self_attn)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i_self_attn)))  # N, dim_proj


        # f_i_self_attn = self.multihead_diffattn_img(f_i_tilde, f_i_tilde)  # N, 
        # e_i_self_attn = self.multihead_diffattn_txt(e_i_tilde, e_i_tilde)  # N, 



        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        
        # ####################
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        # joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)  # N, 2*dim_proj
        ##########################

        # cros_attn1, _ = self.multihead_attn1(query=alpha_v_i, key=f_i_tilde, value=f_i_tilde, need_weights=False)
        # cros_attn2, _ = self.multihead_attn2(query=alpha_e_i, key=e_i_tilde, value=e_i_tilde, need_weights=False)
        

        # print("joint_repr", joint_repr.shape)
        # joint_repr torch.Size([8, 1296, 64])

        # attn_out = self.self_attn_last(masked_e_i, masked_v_i, masked_v_i)
        # attn_out = self.self_attn_last(joint_repr)
        # attn_out = self.self_attn_last(joint_repr, joint_repr)
        attn_out = self.cross_attn_last(masked_e_i, masked_v_i)

        # print("cros_attn1", cros_attn1.shape)
        # cros_attn1 torch.Size([8, 1296, 8])

        # x = self.mlp_head(cros_attn1[:, 0, :])

        # joint_repr = torch.cat((cros_attn1, cros_attn2),dim=1)  # N, 2*dim_proj
        # Get class label prediction logits with final fully-connected layers

        # return x
        return self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(attn_out)))))








class DenseNetBertMMModel_crisiskan(MMModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=1000, dim_text_repr=768, dim_proj=100, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

       
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
       
        #config = BertConfig()
        #textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        #config = AlbertConfig()
        #textEncoder = AlbertModel(config).from_pretrained('albert-base-v2')
        config = ElectraConfig()
        textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        #config = XLNetConfig()
        #textEncoder = XLNetModel(config).from_pretrained('xlnet-base-cased')


        super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder, save_dir)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)
        

    def forward(self, x):
        image, text = x

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr
        
        # Applying self-attention to f_i and e_i
        f_i_self_attn = self.apply_self_attention(f_i)  # N, dim_proj
        e_i_self_attn = self.apply_self_attention(e_i)  # N, dim_proj
        
        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i_self_attn)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i_self_attn)))  # N, dim_proj

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i),
                               dim=1)  # N, 2*dim_proj

        # Get class label prediction logits with final fully-connected layers
        return self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint_repr)))))