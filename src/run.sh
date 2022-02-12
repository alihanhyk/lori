for k in {0..4}
do
    echo "key = $k"
    echo "generating data ..."
    python src/data.py -k $k

    echo "running main-lori.py ..."
    python src/main-lori.py -k $k --silent
    echo "running main-trex.py ..."
    python src/main-trex.py -k $k --silent
    echo "running main-birl.py ..."
    python src/main-birl.py -k $k --silent
    echo "running main-bc.py ..."
    python src/main-bc.py -k $k --silent

    echo "simulating results ..."
    python src/eval-simu1.py
done

echo '---'
python src/eval.py
echo '---'
python src/eval1.py
