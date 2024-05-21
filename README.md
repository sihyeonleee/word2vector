

학습목적의 심플한 프로젝트로 일반적인 테스트는 example 폴더내 cbow.js 와 skipgram.js 를 node 로 실행해보시면 됩니다.
ex) node ./example/skipgram.js 혹은 node ./example/cbow.js

==============================================================================================================================


src/lrn 폴더에 있는 skipgram.js 파일은 좀더 큰 파일을 이용한 테스트로
"src/data/morphemeTarget/[학습파일.txt]"  를 대상으로 src/data/vector/ 하위에 index.txt, vector.txt 가 만들어집니다.

"src/data/morphemeTarget/[학습파일.txt]" 파일은 형태소 분석을 통한 결과 데이터가 필요로 하며 src/lrn/mecabYa.js 형태소 분석기를 사용한 결과 데이터 이며,
파일의 형식은 ["때","디자인","학생","외국","디자이너"] 와 같은 전체 텍스트 대상 1차원 배열로 단어를 나열한 결과 입니다.

mecabYa 는 nori 형태소 분석기에서도 사용한 프로그램으로 windows 에서는 지원하지 않아 리눅스에서 다음과 같이 설치 한후 위 src/lrn/mecabYa.js 를 실행합니다.

 https://peakd.com/kr-dev/@nhj12311/node-and-steem-10-mecab-mecab-ya
 
 $ yum install gcc gcc-c++ make
 $ wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
 $ wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
 $ gzip -d .gz
 $ tar xvf mecab.tar
 
 $ cd mecab-[version]
 $ ./configure
 $ make
 $ make install
 
 $ cd mecab-ko-dic-[version]
 $ ./autogen.sh mecab
 $ ./configure
 $ make
 $ make install
 
 $ npm install mecab-ya
 $ node_modules/mecab-ya/bin/install-mecab ko

형태소 분석 된 학습파일.txt 를 준비 후 node ./src/lrn/skipgram.js 를 실행후 학습시키고, src/data/vector/ 하위 파일들읠 데이터를 활용해 
node ./src/test/test.js 실행시 postman 으로 테스트가 가능합니다.

테스트는 
  3000포트 
  post 요청, /d, /w, 
  json body 에 각 keyword, sentence 입력 해주시면 됩니다.
