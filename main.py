from helper import *
from anfis import anfis

from anfis.anfis import anfis as anf


def main():
    # Load data
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    # Normalize data
    df = normalize_dataframe(df)

    # Define membership functions for each feature
    mf = define_membership_function(df, labels=['DEATH_EVENT'], num_clusters=3, memfunc='gaussmf')
    mfc = anfis.membershipfunction.MemFuncs(mf)

    # Define ANFIS model
    model = anf.ANFIS(df.drop('DEATH_EVENT', axis=1), df['DEATH_EVENT'], mfc)

    # Train model
    model.trainHybridJangOffLine(epochs=20)

    # Plot errors
    model.plotErrors()

    # Plot results
    model.plotResults()


if __name__ == "__main__":
    main()
