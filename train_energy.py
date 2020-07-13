import os
import logging
from torch.optim import AdamW
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError
from schnetpack.train import build_mse_loss
from schnetpack import AtomsData
import schnetpack as spk

cutoff = 5. # Angstrom
environment_provider = spk.environment.AseEnvironmentProvider(cutoff)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
model_dir = "ocp_schnet_model"  # directory that will be created for storing model
os.makedirs(model_dir)
properties = ["energy"]  # properties used for training

# data preparation
logging.info("get dataset")
train_dataset = AtomsData("ocp1k_adslab_train_schnet.db", 
                          load_only=properties,
                          environment_provider=environment_provider,
                         )
val_dataset = AtomsData("ocp1k_adslab_val.db_schnet", 
                        load_only=properties,
                        environment_provider=environment_provider,
                       )


train_loader = spk.AtomsLoader(train_dataset, 
                               batch_size=32,
                              num_workers=1)
val_loader = spk.AtomsLoader(val_dataset, 
                             batch_size=32,
                            num_workers=1)

# get statistics
atomrefs = train_dataset.get_atomref(properties)
per_atom = dict(energy=False)
means, stddevs = train_loader.get_statistics(
    properties, single_atom_ref=atomrefs, divide_by_atoms=per_atom
)

# model build
logging.info("build model")
representation = spk.SchNet(n_interactions=6, )
output_modules = [
    spk.atomistic.Atomwise(
        n_in=representation.n_atom_basis,
        property="energy",
        mean=means["energy"],
        stddev=stddevs["energy"],
        aggregation_mode='avg'
    )
]
model = schnetpack.atomistic.model.AtomisticModel(representation, output_modules)

# build optimizer
optimizer = AdamW(params=model.parameters(), lr=1e-3)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
hooks = [CSVHook(log_path=model_dir, metrics=metrics), ReduceLROnPlateauHook(optimizer)]

# trainer
loss = build_mse_loss(properties, loss_tradeoff=[1])
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# run training
logging.info("training")
trainer.train(device="cuda", n_epochs=1000)
