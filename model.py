import torch
import torch.nn.functional as F
from torch_rechub.basic.layers import MLP, EmbeddingLayer

class YoutubeDNN(torch.nn.Module):
    """The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
    It's a DSSM match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.
    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, item_features, neg_item_feature, user_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # calculate cosine score
        y = torch.mul(user_embedding, item_embedding).sum(dim=2)
        y = y / self.temperature
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, 1, embed_dim]
        user_embedding = F.normalize(user_embedding, p=2, dim=2)
        if self.mode == "user":
            return user_embedding.squeeze(1)  #inference embedding mode -> [batch_size, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        pos_embedding = F.normalize(pos_embedding, p=2, dim=2)
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=2)
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]

#--------------------------------------------------------------------------
class ActivationUnit(nn.Module):
    """Activation Unit Layer mentioned in DIN paper, it is a Target Attention method.

    Args:
        embed_dim (int): the length of embedding vector.
        history (tensor):
    Shape:
        - Input: `(batch_size, seq_length, emb_dim)`
        - Output: `(batch_size, emb_dim)`
    """

    def __init__(self, emb_dim, dims=None, activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        if dims is None:
            dims = [36] # just like the origin paper DIN`
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MLP(4 * self.emb_dim, dims=dims, activation=activation) # output_layer=True is default set

    def forward(self, history, target):
        seq_length = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_length, -1)  #batch_size,seq_length,emb_dim
        att_input = torch.cat([target, history, target - history, target * history],
                              dim=-1)  # batch_size,seq_length,4*emb_dim[p=]
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))  #  #(batch_size*seq_length,4*emb_dim)
        att_weight = att_weight.view(-1, seq_length)  #(batch_size*seq_length, 1) -> (batch_size,seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)
        # (batch_size, seq_length, 1) * (batch_size, seq_length, emb_dim)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1)  #(batch_size,emb_dim)
        return output

class YDNNA(torch.nn.Module):
    """
    It's a model modified from YouTube DNN, which adds attention mechanism in it
    It's a DSSM match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        attention_mlp_params (dict): the params of the attention MLP structure, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, item_features, neg_item_feature, user_params, attention_mlp_params={"dims": [256, 128]}, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature

        self.user_sequence_features = []
        self.user_sparse_features = []
        for fea in user_features:
          if isinstance(fea, SequenceFeature):
            self.user_sequence_features.append(fea)
          if isinstance(fea, SparseFeature):
            self.user_sparse_features.append(fea)
        #for fea in item_features: # as for now, we don't have any data of it
        #  if isinstance(fea, SequenceFeature):
        #    self.item_sequence_features += fea
        self.num_sequence_features = len(self.user_sequence_features)
        self.num_sparse_features = len(self.user_sparse_features)

        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features]) #for MLP input dimension
        self.embedding = EmbeddingLayer(user_features + item_features)

        self.attention_layers = nn.ModuleList([ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.user_sequence_features])

        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params, activation='dice',)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # calculate cosine score
        y = torch.mul(user_embedding, item_embedding).sum(dim=2)
        y = y / self.temperature
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
#------------------------------
#attention mechanism
        user_sparse_embedding = self.embedding(x, self.user_sparse_features)
        user_sequence_embedding = self.embedding(x, self.user_sequence_features)
        pos_item_embedding = self.embedding(x, self.item_features)
        pos_item_embedding = F.normalize(pos_item_embedding, p=2, dim=2)
        attention_pooling = []
        for i in range(self.num_sequence_features):
          attention_seq = self.attention_layers[i](user_sequence_embedding[:, i, :, :], pos_item_embedding[:, i, :])
          attention_pooling.append(attention_seq.unsqueeze(1)) #(batch_size, 1, emb_dim)
        attention_pooling = torch.cat(attention_pooling, dim=1) #(batch_size, num_sequence_features, emb_dim)
        input_user = torch.cat([user_sparse_embedding.flatten(start_dim=1), attention_pooling.flatten(start_dim=1)], dim=1) #(batch_size, N)
#------------------------------
        #input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, 1, embed_dim]
        user_embedding = F.normalize(user_embedding, p=2, dim=2)
        if self.mode == "user":
            return user_embedding.squeeze(1)  #inference embedding mode -> [batch_size, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        pos_embedding = F.normalize(pos_embedding, p=2, dim=2)
        #pos_embedding = pos_embedding.flatten(start_dim=1)
        #pos_embedding = pos_embedding.unsqueeze(dim=1)
        pos_embedding = pos_embedding.mean(dim=1)
        pos_embedding = pos_embedding.unsqueeze(dim=1)
        #print(pos_embedding.size())
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=2)
        #print('neg:', neg_embeddings.size())
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]