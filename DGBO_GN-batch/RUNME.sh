

#for overfit_type in  "PART" "ADD" "NONE" #"All"
#do
    for rseed in 264852 312213 314150 434234 23456 #23456
    do

    #pythonw DGBO.py --run=True --initn=20 --maxiter=180 --rseed=$rseed --max_pending=1 --dataset="delaney-processed"  --resample_period=20 --relearn_NN_period=20 --weight_decay=0.07
    #pythonw DGBO.py --run=True --initn=20 --maxiter=180 --rseed=$rseed --max_pending=1 --dataset="synthetic_datasets"  --resample_period=20 --relearn_NN_period=20 --weight_decay=0.07 --overfit_type=$overfit_type
#pythonw DGBO.py --run=True --initn=20 --maxiter=50 --rseed=$rseed --max_pending=50 --dataset="250k_rndm_zinc_drugs_clean_3"  --resample_period=1 --relearn_NN_period=1 --weight_decay=1e-5 #--topN=20000
        pythonw DGBO.py --run=True --initn=200 --maxiter=15 --rseed=$rseed --max_pending=20 --dataset="20k_rndm_zinc_drugs_clean_3"  --resample_period=1 --relearn_NN_period=1 --weight_decay=1e-5 --topN=20000

    done
#done
