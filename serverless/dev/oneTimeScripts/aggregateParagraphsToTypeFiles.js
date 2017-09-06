const fs = require('fs');

const dataFolderPath = './data/';

fs.readdir(dataFolderPath, (err, files) => {
  for (let file of files){
    let data = fs.readFileSync(dataFolderPath + file, 'utf8');
  
    let dataList = data.split(/\r?\n|\r/);
    // console.log("dataList", dataList);
    let category;
    let content = "";
    
    for (let sentence of dataList) {
      if (sentence.length === 0) continue;
      else if (!category) {
        category = sentence.replace(/[ -]/g, "");
      } else {
        content += sentence;
        // add comma if the sentence doesn't end with any punctuation mark
        content += sentence[sentence.length - 1].match(/[。？，！]/) ? "" : "，"
      }
    }
  
    if (!fs.existsSync("./processedData/")){
      fs.mkdirSync('processedData');
    }
    fs.appendFileSync(`./processedData/${category}.txt`, `${content}\n`);
  }
  console.log("Successfully aggregate files.")
});