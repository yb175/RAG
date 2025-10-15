import readlineSync from "readline-sync";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";
dotenv.config();
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const History = [];

// Context + initial query 
async function aiResponse(context) {
  History.push({
    role: "user",
    parts: [{ text: context }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a chemistry teacher of 11th class and you have knowledge of 
      Some basic concepts of chemistry answer the user query , your tone should be fun , playfull and 
      easily understood by students , if anything is asked apart from context do not answer it just 
      write a user friendly message that this is beyond your scope
      Context: ${context}
      `,
    },
  });

  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });

  console.log("\n");
  console.log(response.text);
}

// Refining the query for hanling follow up questions
async function refineQuery(question) {
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    Only output the rewritten question and nothing else.
      `,
    },
  });

  History.pop();

  return response.text;
}

// thats for refining->finding context from vector db->llm 
async function chatting(userProblem) {
  const refinedQuery = await refineQuery(userProblem) ; 
  const queryVector = await embeddings.embedQuery(refinedQuery);
  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });
  const context = [
    `User query : ${refinedQuery}`,
    ...searchResults.matches.map((match) => match.metadata.text),
  ].join("\n\n---\n\n");
  await aiResponse(context);
}


async function main() {
  const userProblem = readlineSync.question("Ask me anything--> ");
  await chatting(userProblem);
  await main();
}

main();
