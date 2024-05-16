/**
 * window 사용 불가
 * centos 셋팅후 사용
 * https://peakd.com/kr-dev/@nhj12311/node-and-steem-10-mecab-mecab-ya
 * 
 * $ yum install gcc gcc-c++ make
 * $ wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
 * $ wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
 * $ gzip -d .gz
 * $ tar xvf mecab.tar
 * 
 * $ cd mecab-[version]
 * $ ./configure
 * $ make
 * $ make install
 * 
 * $ cd mecab-ko-dic-[version]
 * $ ./autogen.sh mecab
 * $ ./configure
 * $ make
 * $ make install
 * 
 * $ npm install mecab-ya
 * $ node_modules/mecab-ya/bin/install-mecab ko
 */

const fs = require('fs');
const mecab = require('mecab-ya');

const readFileName = 'text.txt';
const f = fs.readFileSync(readFileName);
const sentences = f.toString();

const writeFileName = 'resule.txt';

// 전체문서 한번에 분류
async function analyzeAndSave() {

    let sentencesSplit = sentences.split('.');

    let results = [];
    let batchSentences = [];

    for (let i = 0; i < sentencesSplit.length; i++) {
        batchSentences.push(sentencesSplit[i]);

        if ((i + 1) % 1000 === 0 || i === sentencesSplit.length - 1) {
            console.log(`${i+1}/${sentencesSplit.length}`);
            const nouns = await new Promise((resolve, reject) => {
                mecab.nouns(batchSentences.join('. '), (err, result) => {
                    if (err) {
                        console.error('Error!');
                        console.error(err);
                        reject(err);
                    } else {
                        resolve(result);
                    }
                });
            });
            results = [...results, ...nouns];
            batchSentences = [];  // Reset the batch sentences
        }
    }


    fs.writeFileSync(`./${writeFileName}`, JSON.stringify(results));

}


// 특정 기준으로 분류 하여 2중 배열 분류
async function analyzeAndSave2() {
    let sentencesSplit = sentences.split('\n');
    let resultCnt = 0;
    let results = [];
    let promises = [];
    for (let sentence of sentencesSplit) {
        let promise = new Promise((resolve, reject) => {
            mecab.nouns(sentence, (err, result) => {
                if (err) {
                    console.error('Error!', err);
                    reject(err);
                } else {
                    resolve(result);
                }
            });
        });
        promises.push(promise);

        if(resultCnt % 1000 == 0 || resultCnt === sentencesSplit.length - 1) {
            console.log(`${resultCnt+1} / ${sentencesSplit.length}`);
            results = [...results, ...await Promise.all(promises)];
            promises = [];
        }
        resultCnt++;
    }

    try {
        // 모든 프로미스가 완료될 때까지 기다린 후 결과를 받음
        // 결과를 파일에 JSON 형태로 저장
        fs.writeFileSync(`./${writeFileName}`, JSON.stringify(results), 'utf8');
        console.log('All sentences have been analyzed and saved.');
    } catch (err) {
        console.error('An error occurred during analysis or file saving:', err);
    }
}

analyzeAndSave2();
