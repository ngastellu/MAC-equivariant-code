#!/usr/bin/env python

from pathlib import Path 
from check_losses import parse_multiple_losses, plot_losses


graphene_logs_path = Path("~/Desktop/simulation_outputs/equivariant_MAC/rot_90_conv_layer_60_graphene_logs").expanduser()
logs = list(graphene_logs_path.iterdir())
epochs, tr_loss, te_loss = parse_multiple_losses(logs)
plot_losses(epochs, tr_loss,te_loss)

epoch1 = 2000
epoch2 = 3000
ind1 = (epochs == epoch1).nonzero()[0]
ind2 = (epochs == epoch2).nonzero()[0]

print(f'Training loss at epoch {epoch1} < Training loss at epoch {epoch2}:  {tr_loss[ind1] < tr_loss[ind2]}')
print(f'Test loss at epoch {epoch1} < Test loss at epoch {epoch2}: {te_loss[ind1] < te_loss[ind2]}')