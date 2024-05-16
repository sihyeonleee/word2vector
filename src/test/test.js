
const fs = require('fs');
const express = require('express');
const app = express();
const port = 3000;

app.use(express.json()); 
    
app.post('/w', (req, res) => {
    let result = wordSimilarity(req.body.keyword);
    res.send(result);
});

app.post('/d', (req, res) => {
    let result = docSimilarity(req.body.word);
    res.send(result);
});

app.listen(port, () => {
    console.log(`server is listening at localhost:${port}`);
});





const dataPath = './src/data/vector/';
const vectorFileName = 'vector.txt';
const indexFileName = 'index.txt';;


const wordWeights = JSON.parse(fs.readFileSync(dataPath + vectorFileName).toString());
const vocabulary = JSON.parse(fs.readFileSync(dataPath + indexFileName).toString());



function wordSimilarity(testWord){
    
    let similarityList = [];
    let similarity = -Infinity;
    let similarIndex = 0;
    let similarWord = '';
    let trgetIndex = vocabulary.indexOf(testWord);
    console.log(`테스트 단어! ::: ${vocabulary[trgetIndex]} / 위치 ::: ${trgetIndex}`);
    if(trgetIndex < 1) {
        console.log('매칭불가 :: 신규 단어 재학습 필요')
        return '매칭불가 :: 신규 단어 재학습 필요';
    }
    for (var w=0; w<vocabulary.length; w++) {
        if (w !== trgetIndex) {
            similarity = cosineSimilarity(wordWeights[w], wordWeights[trgetIndex]);
            // console.log(`similarWord ${vocabulary[w]}`);
            // console.log(`similarity ${similarity}`);
            similarWord = vocabulary[w];
            similarIndex = w;
            similarityList.push({
                similarity : similarity,
                similarWord : similarWord,
                similarIndex : similarIndex,
                msg : `유사 단어 : ${similarWord} / 유사 단어 위치 : ${similarIndex} / 유사도 : ${similarity}`
            });
        }
    }

    similarityList.sort((a,b) => b.similarity - a.similarity);

    return { 
        result : similarityList.slice(0, 10),
        input : `테스트 단어! ::: ${vocabulary[trgetIndex]} / 위치 ::: ${trgetIndex}`
    };

}


// 문자열 벡터 평균
function documentVectorAvg(document, weights, vocabulary) {
    const words = document.split(/\s+/);
    const docVector = new Array(weights[0].length).fill(0);
    let validWords = 0;

    words.forEach(word => {
        let wordIndex = vocabulary.indexOf(word);
        if(wordIndex > -1){
            validWords++;
            const wordEmbedding = weights[wordIndex];
            wordEmbedding.forEach((value, index) => {
                docVector[index] += value;
            });
        }
    });

    return validWords > 0 ? docVector.map(value => value / validWords) : docVector;
}


function docSimilarity(testDoc) {

    const sentences = testDoc.split('.');  // .(dot) 으로 문장 분리
    const docVector = documentVectorAvg(testDoc, wordWeights, vocabulary);  // 문서 벡터
    const summaries = [];

    // console.log(sentences.length);

    sentences.forEach(sentence => {
        if(sentence.trim() == '') return;
        const sentenceVector = documentVectorAvg(sentence, wordWeights, vocabulary);  // 문장 벡터
        const similarity = cosineSimilarity(docVector, sentenceVector); 
        console.log(`similarity :: ${similarity} // sentence :: ${sentence}`);
        if (similarity > 0.5) {  // 설정된 임계값(0.5) 이상의 유사도를 가진 문장만을 요약으로 사용
            summaries.push(sentence);
        }
    });

    console.log(`result :: ${summaries.join(', ')}`);
    return summaries.join(', ');

}


// 코사인 유사도
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
