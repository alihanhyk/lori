echo "processing raw data ..."
python src/data-liver.py

echo "running main-lori-liver.py ..."
python src/main-lori-liver.py --silent
echo "running main-trex-liver.py ..."
python src/main-trex-liver.py --silent

echo "---"
python src/eval-liver.py
