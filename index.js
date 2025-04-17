import express from 'express';
import multer from 'multer';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import fs from 'fs'; // Using fs.promises for async file operations
import cors from 'cors';
import path from 'path';
import os from 'os'; // For temporary directory
import axios from 'axios';
import FormData from 'form-data';
// Import dayjs and necessary plugins for time resolution
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime.js'; // Ensure .js
import localizedFormat from 'dayjs/plugin/localizedFormat.js'; // Ensure .js
import utc from 'dayjs/plugin/utc.js'; // Ensure .js
import timezone from 'dayjs/plugin/timezone.js'; // Ensure .js

// Extend dayjs with plugins
dayjs.extend(relativeTime);
dayjs.extend(localizedFormat);
dayjs.extend(utc);
dayjs.extend(timezone);

// Load environment variables from .env file
dotenv.config();

// --- Configuration ---
const app = express();
const port = 3001; // Or your preferred port

// Enable CORS for all origins (adjust for production later)
app.use(cors());

// Check for OpenAI API Key
if (!process.env.OPENAI_API_KEY) {
  console.error('ERROR: OPENAI_API_KEY environment variable is not set.');
  process.exit(1); // Exit if the key is missing
}

// Initialize OpenAI client (still needed for GPT-4 call)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 60 * 1000, // 60 seconds timeout
});

// Configure Multer for in-memory file storage
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// --- API Endpoint ---

/**
 * Endpoint to process uploaded audio for baby tracking logs.
 * Expects a multipart/form-data request with an 'audio' field containing the audio file.
 */
app.post('/process-audio-log', upload.single('audio'), async (req, res) => {
  console.log('Received request to /process-audio-log');

  // Validate File Upload
  if (!req.file) {
    console.error('No audio file uploaded.');
    return res.status(400).json({ error: 'No audio file uploaded.' });
  }
  console.log(`Uploaded file: ${req.file.originalname}, size: ${req.file.size}, mimetype: ${req.file.mimetype}`);

  let transcriptionText = "Transcription failed."; // Default if transcription fails
  let tempFilePath = null; // Variable to hold the temp file path for cleanup

  try {
    // --- 1. Save Buffer to Temporary File ---
    // Create a unique temporary filename
    const tempFileName = `upload_${Date.now()}_${path.basename(req.file.originalname || 'audio.tmp')}`; // Use path.basename for safety
    tempFilePath = path.join(os.tmpdir(), tempFileName);

    console.log(`Saving buffer to temporary file: ${tempFilePath}`);
    await fs.promises.writeFile(tempFilePath, req.file.buffer);

    // --- 2. Transcribe Audio with Whisper using Axios and FormData ---
    console.log('Sending audio file stream via Axios to OpenAI Whisper...');

    const whisperFormData = new FormData();
    // IMPORTANT: Use fs.createReadStream to stream the temp file
    whisperFormData.append('file', fs.createReadStream(tempFilePath));
    whisperFormData.append('model', 'whisper-1');
    whisperFormData.append('response_format', 'text');
    // whisperFormData.append('language', 'en'); // Optional: Specify language

    const whisperApiUrl = 'https://api.openai.com/v1/audio/transcriptions';

    const axiosConfig = {
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        ...whisperFormData.getHeaders(), // Let FormData set the Content-Type with boundary
      },
      timeout: 60 * 1000, // 60 second timeout for axios request
    };

    const response = await axios.post(whisperApiUrl, whisperFormData, axiosConfig);

    transcriptionText = response.data; // Assign successful transcription
    console.log('Transcription received:', transcriptionText);

    // --- 3. Extract Structured Data with GPT-4 ---
    console.log('Sending transcription to GPT-4 for data extraction...');

    // Get current time for relative resolution, guessing user's timezone
    const now = dayjs();
    const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'America/Chicago'; // Fallback timezone
    const currentTimeString = now.tz(userTimezone).format('llll z'); // e.g., "Thu, Apr 17, 2025 10:30 AM CDT"

    // *** v13 SYSTEM PROMPT ***
    const systemPrompt = `
You are an assistant helping parents log baby activities from transcribed voice notes. The current time is ${currentTimeString}. Extract key information into a JSON object.

JSON Fields:
- "babyName": string | null
- "event": string | null - Categorize: "diaper change", "feed", "sleep", "pump", "medication", "temperature", "bath", "note".
- "time": string | null - **CRITICAL:** Resolve relative times ("right now", "1 hour ago", "last night at 10", "this morning") to a specific clock time string (HH:mm AM/PM format) based on the current time (${currentTimeString}). Do NOT return the relative phrase (e.g., "an hour ago"). If a specific time like "8 AM" is mentioned, use that directly. Example: If current time is 5:05 PM CDT and user says "fed an hour ago", return "4:05 PM". If user says "pooped right now", return "5:05 PM". Default null ONLY if no time reference is made.
- "details": object - Specifics:
    - "diaper change": { "type": "poop" | "pee" | "poop and pee" | "dry" | null, "consistency": "runny" | "soft" | "formed" | "hard" | "watery" | null, "color": "yellow" | "brown" | "green" | "black" | "red" | "white" | null }
    - "feed": { "type": "breast" | "bottle" | "solids" | null, "milkType": "formula" | "breast milk" | null (for bottle/breast), "amount": string | null (e.g., "120", "5"), "unit": "oz" | "ml" | null, "food": string | null (for solids), "duration": string | null (for breast, e.g., "15 min L / 10 min R") }
    - "sleep": { "type": "nap" | "night" | null, "duration": string | null, "location": string | null }
    - "pump": { "amount": string | null (e.g., "150", "4.5"), "unit": "oz" | "ml" | null, "duration": string | null }
    - "medication": { "name": string | null, "dosage": string | null }
    - "temperature": { "value": string | null (e.g., "37.5", "98.6"), "unit": "C" | "F" | null }
    - Others: {}
- "promptForDetails": string[] | null - **CRITICAL:** List keys of essential details that were NOT mentioned OR were mentioned ambiguously (e.g., 'some milk'). Check the 'details' object *after* initial extraction. Only list if the value is NULL.
    - For event="diaper change" AND details.type includes "poop": If details.consistency IS null OR details.color IS null, add "consistency", "color".
    - For event="feed": If details.type IS null OR (details.type === 'bottle' AND details.amount IS null), add "type", "amount". If details.type IS "bottle" AND details.milkType IS null, add "milkType". If details.type IS "breast" AND details.duration IS null, add "duration". If details.type IS "solids" AND details.food IS null, add "food".
    - For event="pump": If details.amount IS null, add "amount".
    - For event="temperature": If details.value IS null, add "value".
    - If no details are missing based on these rules, return null. Ensure the list contains unique keys and is not empty if details are missing.
- "originalTranscription": string

Rules:
- Return ONLY the JSON object.
- Default unspecified values to null.
- **TIME RESOLUTION IS MANDATORY.** Convert relative times to HH:mm AM/PM based on current time: ${currentTimeString}. Do not output phrases like "an hour ago". Example: Input: "fed 2 hours ago" at 5:00 PM -> Output time: "3:00 PM". Input: "pumped right now" at 10:15 AM -> Output time: "10:15 AM".
- Infer 'bottle' feed type if milk/formula mentioned without 'breast'/'nursing'. Extract 'milkType' if mentioned.
- **Prompting:** Be diligent in checking for missing essential details based on the event type and add the corresponding keys to 'promptForDetails'. If the user says something vague like "fed the baby", 'type' and 'amount' should likely be in 'promptForDetails'.
`;

    const chatCompletion = await openai.chat.completions.create({
        model: 'gpt-4-turbo',
        messages: [ { role: 'system', content: systemPrompt }, { role: 'user', content: transcriptionText } ],
        response_format: { type: 'json_object' },
    });

    const extractedDataString = chatCompletion.choices[0]?.message?.content;
    console.log('Raw data extracted by GPT:', extractedDataString);
    if (!extractedDataString) throw new Error('GPT-4 did not return content.');

    // --- 4. Parse and Validate Extracted JSON ---
    let structuredData;
    try {
      structuredData = JSON.parse(extractedDataString);
      // Ensure originalTranscription is always included, even if GPT forgets
      structuredData.originalTranscription = transcriptionText;
       // Ensure promptForDetails is null or a non-empty array of strings
      if (structuredData.promptForDetails && (!Array.isArray(structuredData.promptForDetails) || structuredData.promptForDetails.length === 0)) {
          structuredData.promptForDetails = null;
      }
      console.log('Structured data parsed:', structuredData);
    } catch (parseError) {
        console.error('Failed to parse JSON from GPT:', parseError);
        // Fallback structure
        structuredData = {
            babyName: null, event: "note", time: null, details: {},
            originalTranscription: transcriptionText,
            error: "AI could not structure the data reliably.",
            promptForDetails: null
        };
    }

    // --- 5. Send Structured Data to Client ---
    res.status(200).json(structuredData);

  } catch (error) { // Catch any error
        console.error('Error processing audio log:', error);
        let status = 500;
        let message = 'Failed to process audio log.';
        let details = error.message || 'An unknown error occurred';

        // Check for specific error types to provide better feedback
        if (axios.isAxiosError(error)) {
            console.error('Axios Error Status:', error.response?.status);
            console.error('Axios Error Response:', error.response?.data);
            status = error.response?.status || 500;
            details = error.response?.data?.error?.message || details;
            message = `Backend API call failed: ${status}`;
        } else if (error instanceof OpenAI.APIError) {
            console.error('OpenAI API Error Status:', error.status);
            console.error('OpenAI API Error Message:', error.message);
            status = error.status || 500;
            details = error.message || details;
            message = `OpenAI API error: ${status}`;
        } else {
            console.error('Non-API Error:', error.message);
        }
        res.status(status).json({
            error: message,
            details: details,
            // Include transcription if it succeeded before the error
            transcription: (transcriptionText !== "Transcription failed.") ? transcriptionText : null,
        });
  } finally {
    // --- 6. Clean up Temporary File ---
    if (tempFilePath) {
        console.log(`Cleaning up temporary file: ${tempFilePath}`);
        try {
            // Use asynchronous unlink and don't necessarily wait for it
            // fs.promises.unlink(tempFilePath).catch(cleanupError => {
            //     console.error(`Non-blocking: Failed to delete temp file ${tempFilePath}:`, cleanupError);
            // });
            // OR use synchronous if cleanup must happen before response (less ideal)
            // fs.unlinkSync(tempFilePath);
            // Let's stick with async await for simplicity here
             await fs.promises.unlink(tempFilePath);
             console.log('Temporary file deleted successfully.');
        } catch (cleanupError) {
            console.error(`Failed to delete temporary file ${tempFilePath}:`, cleanupError);
        }
    }
  }
});

// Basic root endpoint
app.get('/', (req, res) => {
  res.send('NurtureTrack Backend is running!');
});

// Start the server
app.listen(port, () => {
  console.log(`NurtureTrack backend server listening on http://localhost:${port}`);
});
