// Now all the exports would be imported not only default ones 
import * as dotenv from "dotenv"
dotenv.config() 
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

// Loading the pdf 
const PDFPath = './ch1.pdf' 
const loader = new PDFLoader(PDFPath) ; 
const rawDocs = await loader.load();

// console.log(rawDocs.length)

// Chunking the pdf 
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200, // number of words 
    chunkOverlap: 50, // overlaping allowed 
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

// console.log(JSON.stringify(chunkedDocs.slice(0, 2), null, 2));

// Initialising the vector embedding 
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// At one time only 5 chuncks would be proccesed and inserted into db rest would wait 
await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });