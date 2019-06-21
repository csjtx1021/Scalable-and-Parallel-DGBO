
for rseed in 264852 312213 314150 434234 23456 #23456
do
# pythonw DGBO.py --run=True --dataset="250k_rndm_zinc_drugs_clean_3" --maxiter=50 --rseed=$rseed --overfit_type="heur" --dropout=0.279 --weight_decay=1e-5 --max_pending=50 --initn=200 --resample_period=1 --relearn_NN_period=1
    pythonw DGBO.py --run=True --dataset="20k_rndm_zinc_drugs_clean_3" --maxiter=15 --rseed=$rseed --overfit_type="heur" --dropout=0.279 --weight_decay=1e-5 --max_pending=20 --initn=200 --resample_period=1 --relearn_NN_period=1
#pythonw DGBO.py --run=True --maxiter=50 --rseed=$rseed --overfit_type="heur" --dropout=0.279 --weight_decay=1e-5 --max_pending=50 --initn=20 --resample_period=1 --relearn_NN_period=1
#pythonw DGBO.py --run=True --maxiter=180 --rseed=$rseed --overfit_type="heur" --dropout=0.279 --weight_decay=0.1 --max_pending=1
#pythonw DGBO.py --run=True --maxiter=180 --rseed=$rseed --overfit_type="fixed" --max_pending=1 --dropout=0.03 --weight_decay=0.07
#pythonw DGBO.py --run=True --maxiter=180 --rseed=$rseed --overfit_type="heur" --max_pending=1 --dropout=0.279 --weight_decay=0.071 --resample_period=20 --relearn_NN_period=20
done

