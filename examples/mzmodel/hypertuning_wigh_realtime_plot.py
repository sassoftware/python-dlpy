import sys
import swat as sw
from dlpy.mzmodel import *

if __name__ == '__main__':
    s = sw.CAS('your-server.unx.company.com', port)

    s.table.addcaslib(activeonadd=False,
                      datasource={'srctype': 'path'},
                      name='dnfs',
                      path='/example/path',
                      subdirectories=True)

    s.table.loadTable(caslib='dnfs', path='data/cifar10_small.sashdat',
                      casout=dict(name='data', blocksize='1', replace=True))

    model = MZModel(conn=s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                    num_classes=10, model_path="/path/to/resnet18.pt")

    model.add_image_transformation(image_size='28 28')

    lr = HyperRange(lower=5e-4, upper=1e-3)
    batch_size = BatchSizeRange(lower=120, upper=150)

    optimizer = Optimizer(seed=54321,
                          algorithm=SGDSolver(lr=lr, momentum=0.9),
                          batch_size=batch_size,
                          max_epochs=5
                          )

    tuner = Tuner(fidelity=True,
                  fidelity_cut_rate=0.2,
                  fidelity_start_epochs=2,
                  fidelity_step_epochs=2,
                  seed=12345,
                  pop_size=5,
                  fidelity_num_samples=5
                  )

    model.train(table="data",
                inputs="_image_",
                targets="labels",
                gpu=[0],
                tuner=tuner,
                optimizer=optimizer,
                index_variable='labels',
                show_plot=True)



