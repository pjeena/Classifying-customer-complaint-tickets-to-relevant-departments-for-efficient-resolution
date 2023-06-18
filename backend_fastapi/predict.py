import pickle
import re
from pathlib import Path
import numpy as np


def predict_pipeline(text,preprocessor,model,dict_classes):
    #    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    #    text = re.sub(r"[[]]", " ", text)
    #    text = text.lower()
    X_test = preprocessor.transform([text])
    pred = model.predict(X_test)
    complaint_type = dict_classes[str(pred[0])]

    # converting distances to class probabilities
    predict_proba_dist = model.decision_function(X_test)[0]
    e_x = np.exp( predict_proba_dist - np.max(predict_proba_dist) )
    predict_proba = e_x/e_x.sum()

    return complaint_type, list(predict_proba)


#if __name__ == "__main__":
#    config = read_yaml_file()
#    model = load(config["model_trainer"]["model_path"])
#    preprocessor = pickle.load(
#        open(config["data_transformation"]["preprocessor_path"], "rb")
#    )
#    dict_classes = pd.read_csv(
#        config["data_transformation"]["labels_mapping"], index_col=0
#    )

#    text =  "I applied for student loan forgiveness on the form sent to me for such purpose by Sallie Mae. \nI sent the completed application to the proper address per the instructions given by Sallie Mae for XXXX unemployable veterans because of a XXXX service connected XXXX. Sallie Mae 's instructions for XXXX veterans that came with the form state that no physician 's statement is necessary to be completed in the case of service connected XXXX veterans. However, the only part of the application that I did not send to Sallie Mae via certified mail was a physician 's statement. I completed every line of the application other than the physician 's statement. Sallie Mae wrote back in response to my application that it could not review my application at this time because my application was incomplete. My first payment is due in four days which is not enough time to resubmit another application no different than the one I have already submitted. Please help me."

#    print('--------'  + predict_pipeline(text,preprocessor,model,dict_classes) + '-------------')
#    print(type(predict_pipeline(text,preprocessor,model,dict_classes)))
