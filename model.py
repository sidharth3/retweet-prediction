class DNN(nn.Module):
    '''
    ref: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/core.py
    '''
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [nn.ReLU(inplace=True) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

class MLP(nn.Module):
    def __init__(
        self,
        dnn_input, dnn_hidden_units, dnn_dropout,
        activation='relu', use_bn=True, l2_reg=1e-4, init_std=1e-4,
        device='cpu',
        feature_index={},
        embedding_dict={},
        dense_features=[],
        sparse_features=[],
        varlen_sparse_features=[],
        varlen_mode_list=[],
        embedding_size=8,
        batch_size=256,
    ):
        super().__init__()
        self.device = device
        self.feature_index = feature_index
        self.embedding_dict = embedding_dict
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.varlen_sparse_features = varlen_sparse_features
        self.varlen_mode_list = varlen_mode_list
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.reg_loss = torch.zeros((1,), device=device)

        self.dnn = DNN(
            dnn_input, dnn_hidden_units,
            activation='relu', l2_reg=l2_reg, dropout_rate=dnn_dropout, use_bn=use_bn,
            init_std=init_std, device=device
        )

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        # add regularization
        self.add_regularization_loss(self.embedding_dict.parameters(), l2_reg)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg)

        self.out = nn.Sigmoid()

        self.to(device)

    def forward(self, X):
        #print(self.feature_index)
        dense_value_list = [
            X[:, self.feature_index[feat]: self.feature_index[feat] + 1] for feat in self.dense_features
        ]
        
        
        sparse_embedding_list = [
            self.embedding_dict[feat](
                X[:, self.feature_index[feat]: self.feature_index[feat] + 1].long()
            ) for feat in self.sparse_features
        ]
        #ERROR IS HERE
        varlen_sparse_embedding_list = get_varlen_pooling_list(
            self.embedding_dict, X, self.feature_index, self.varlen_sparse_features, self.varlen_mode_list, self.device
        )

        sparse_embedding_list = sparse_embedding_list + varlen_sparse_embedding_list
       # print('here2')
        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
       # print('here3')
        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        return y_pred

    def predict(self, x, batch_size=256):

        model = self.eval()
        test_loader = SimpleDataLoader(
            [torch.from_numpy(x.values)],
            batch_size=batch_size,
            shuffle=False
        )

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).squeeze()
                pred_ans.append(y_pred.cpu().detach().numpy())

        return np.concatenate(pred_ans)

    def add_regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = torch.zeros((1,), device=self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=p, )
            else:
                l2_reg = torch.norm(w, p=p, )
            reg_loss = reg_loss + l2_reg
        reg_loss = reg_loss * weight_decay
        self.reg_loss = self.reg_loss + reg_loss.item()