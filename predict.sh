py=/home/petrmiculek/Code/asdl/env/bin/python

model_weights=shared_resources/model_weights2.pt
data=shared_resources/real_test_for_milestone3/real_inconsistent.json
dest=shared_resources/predictions_inc.json
$py milestone_2/Predict.py --model ${model_weights} --source ${data} --destination ${dest}
