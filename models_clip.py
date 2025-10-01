import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from transformers import BertModel, BertConfig, AlbertModel, AlbertConfig, XLNetConfig, XLNetModel, ElectraConfig, ElectraModel
import os
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from transformers import CLIPModel, AutoConfig, AutoModel
import copy
from diff_transformer import MultiheadDiffAttn


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


class CLIP_CrisiKAN(BaseModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=768, dim_text_repr=768, dim_proj=128, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        
        super(CLIP_CrisiKAN, self).__init__(save_dir)



        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        self.map_dim = dim_proj


        self.dropout = Dropout()

        self.image_map = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_visual_repr)
            )
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_text_repr)
            )

        
        for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip



        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        self.layer_attn_visual = nn.Linear(self.dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(self.dim_text_repr, dim_proj)


        
        
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)






    def forward(self, x):
        # image, text = x
        image, text = x


        image_features = self.image_encoder(pixel_values=image).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(**text).pooler_output
        text_features = self.text_map(text_features)


        
        f_i_self_attn = self.apply_self_attention(image_features)  
        e_i_self_attn = self.apply_self_attention(text_features)  

        
        # # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(image_features)))  
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(text_features)))  

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        
        # ####################
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)  # N, 2*dim_proj
        ##########################

        


        return self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint_repr )))))




class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_weights = nn.Linear(embed_dim, 1)  # Learnable weights

    def forward(self, x):
        attn_scores = self.attn_weights(x).squeeze(-1)  # Shape: [batch_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # Weighted sum






class CLIPDiffModel(BaseModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, save_dir, dim_visual_repr=768, dim_text_repr=768, dim_proj=128, num_class=2):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        
        super(CLIPDiffModel, self).__init__(save_dir)

       

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        self.map_dim = dim_proj


        self.dropout = Dropout()

        self.image_map = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_visual_repr)
            )
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_text_repr)
            )

        
        for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip




        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(self.dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(self.dim_text_repr, dim_proj)

           
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)


        
        
        self.multihead_diffattn = MultiheadDiffAttn(embed_dim=2*dim_proj, depth=1, num_heads=4)





    def forward(self, x):
        image, text = x



        text.pop('token_type_ids', None)  





        image_features = self.image_encoder(pixel_values=image).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(**text).pooler_output
        text_features = self.text_map(text_features)


        
        f_i_self_attn = self.apply_self_attention(image_features)  
        e_i_self_attn = self.apply_self_attention(text_features)  

      
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(image_features)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(text_features)))  # N, dim_proj

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        
        # ####################
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)  # N, 2*dim_proj
        ##########################



        cros_attn1 = self.multihead_diffattn(joint_repr, joint_repr)


        return self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(cros_attn1)))))







