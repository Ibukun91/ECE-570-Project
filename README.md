# ECE-570-Project

This is the revised experimental toolkit utilizing LSTM as LSTM-AES refined from the RNSVerify toolkit 
built for the paper "Verification of RNN-Based Neural Agent Environment Systems."
1) HOW TO RUN THE EXPERIMENTS
-To run the experiments for Table 2 in the Term Paper report, simply run
the 'Paper_LSTM_Implementation.ipynb' notebook on Google Colab. This jupyter notebook file invokes
the 'multi_step_pendulum.py' script which also invokes the verify, keras_lstm_checker,lstm_abstractor,models, 
and pendulum .py files.

-To sucessfully run the ipynb file, first ensure all the zipped files are in the same directory on Google Drive.
Then you can edit the second code cell to the directory of your choice. Then you can run the jupyter notebook file. 
By default, the result shown is the Input on Demand(IOD) unrolling method for 4 steps with an initial state angle of -pi/70. 

-To obtain other experimental values shown on Table 2;
Open the 'multi_step_pendulum.py' file on google drive and edit the following lines to obtain for example
the verification time (seconds) with corresponding constraints and variables for the Input on Start (IOS) method,
for 3 steps at an initial state angle of -pi/30: 

---On Line 23, type '3' as default
---On Line 24, type "start" as default
---On Line 25, type 30 as default.

-Save the 'multi_step_pendulum.py' file to drive and run the 'Paper_LSTM_Implementation.ipynb' 
notebook on Colab to obtain the result.

2)
a&b)Which codes were copied from other repositories and references to those repositories.
--Repository link: https://www.doc.ic.ac.uk/download/?package=rns-verify

Codes obtained from repository:
--multi_step_pendulum.py (Made modifications to all gurobi syntaxes in the file (10-15 lines) 
and edited for lstm-related imports.
--keras_lstm_checker (Made 3-5 lines modification to import lstm module and edit some functions.
--lstm_abstractor (Made modifications to create the LSTMAbstractor class)
--ffnn.py
--rnn.py
--cos.h5
--sin.h5
--verify.py
--constants.py
--models.py

c)Codes developed by me:
--lstm_models.py (LSTM cell structure class and build_model function for LSTM model)
--lstm.py (LSTM() model class)
--lstm_weights_generation.ipynb (to generate weight file loaded onto the lstm model on Line 41
in the 'multi_step_pendulum.py' file

3)The files used were the cos.h5, sin.h5, and the weights_pendulum_rnn.h5 file. They are not datasets as 
they are very small files (less than 60kB) and must be uploaded to sucessfully run the code
so I am uploading them. No training was needed for my project as 
I only needed to run simply stateful architectures.
