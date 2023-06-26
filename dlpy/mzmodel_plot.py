import swat as sw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


def animate(hostname=None, port=None):
    tablename = "HyperTuneStatus"
    s = sw.CAS(hostname, port)

    def _animate(i):
        plt.cla()
        r = s.tableexists(name=tablename, caslib="casuser")["exists"]
        if r == 0:
            return

        r = s.summary(dict(name=tablename, caslib="casuser"), input="ParameterSetID")["Summary"]
        p = pd.DataFrame(r)
        n = p["N"]

        maxid = p["Max"][0]
        maxepoch = 0

        if np.isnan(maxid):
            return

        df = \
            s.fetch(sortby=[{'name': 'CumulativeEpoch', 'order': 'DESCENDING'}, {'name': 'Loss', 'order': 'ASCENDING'}],
                    table={'name': tablename, 'vars': [{'name': 'Loss'}, {'name': 'ParameterSetID'}]}, to=3)['Fetch'];
        label1 = df["ParameterSetID"][0]
        label2 = -1
        label3 = -1
        if int(n) > 1:
            label2 = df["ParameterSetID"][1]
        if int(n) > 2:
            label3 = df["ParameterSetID"][2]

        for id in range(int(maxid) + 1):
            df = s.fetch(table=dict(name=tablename, where="ParameterSetID=" + str(id)), sortby="CumulativeEpoch")[
                'Fetch'];
            history = pd.DataFrame(df)

            y = history["Loss"]
            x = history["CumulativeEpoch"]

            if id == label1 or id == label2 or id == label3:
                plt.plot(x, y, label="ParameterSetID: " + str(id))
            else:
                plt.plot(x, y)
            if x[x.size - 1] > maxepoch:
                maxepoch = x[x.size - 1]

        plt.xticks(np.arange(0, maxepoch + 1, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    ani = FuncAnimation(plt.gcf(), _animate, interval=2000)

    plt.tight_layout()
    plt.show()