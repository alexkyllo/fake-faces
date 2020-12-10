source .env


cp -a $FAKE_FACES_DIR/aligned/train/fake/. $COMBINED_DIR_125/train/fake/.
cp -a $FAKE_FACES_DIR/aligned/train/real/. $COMBINED_DIR_125/train/real/.
cp -a $FAKE_FACES_DIR/aligned/valid/fake/. $COMBINED_DIR_125/valid/fake/.
cp -a $FAKE_FACES_DIR/aligned/valid/real/. $COMBINED_DIR_125/valid/real/.
cp -a $FAKE_FACES_DIR/aligned/test/fake/. $COMBINED_DIR_125/test/fake/.
cp -a $FAKE_FACES_DIR/aligned/test/real/. $COMBINED_DIR_125/test/real/.

cp -a $FAIR_FACES_DIR/aligned/train/fake/. $COMBINED_DIR_125/train/fake/.
cp -a $FAIR_FACES_DIR/aligned/train/real/. $COMBINED_DIR_125/train/real/.
cp -a $FAIR_FACES_DIR/aligned/valid/fake/. $COMBINED_DIR_125/valid/fake/.
cp -a $FAIR_FACES_DIR/aligned/valid/real/. $COMBINED_DIR_125/valid/real/.
cp -a $FAIR_FACES_DIR/aligned/test/fake/. $COMBINED_DIR_125/test/fake/.
cp -a $FAIR_FACES_DIR/aligned/test/real/. $COMBINED_DIR_125/test/real/.
