import pandas as pd
import scipy.io as sio

# Import sas dataset of gold.sas7bat
sas = pd.read_sas('2.2 gold.sas7bdat', 'sas7bdat')
print(sas)

# Create hdf file of abc.h5
store = pd.HDFStore('abc.h5')

# Show the hdf file
print(store)

# Save the sas file in the hdf file
store['sas'] = sas

# Show the sas in the hdf file
print(store)

# Close the hdf file
store.close()

# Create mat file of abc.mat and store sas file in it
mat = sio.savemat('abc.mat', {'Gold': sas.to_records()})

# Show the content of the mat file
print(sio.loadmat('abc.mat'))

