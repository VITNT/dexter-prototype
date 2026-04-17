module.exports = async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method Not Allowed" });
    }

    try {
        const glossInput = req.body.aslString;

        if (!glossInput) {
            return res.status(400).json({ error: "Missing ASL string" });
        }

        const API_KEY = process.env.GEMINI_API_KEY;

        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key=${API_KEY}`;

        // Keep the prompt tight — Gemma tends to ramble if you give it room
        const prompt = `You are an ASL gloss-to-English translator.

ASL gloss is a written representation of American Sign Language where each word is an English label for a specific sign. The words are NOT foreign language words — they are literal English names for hand signs. Do NOT try to interpret or translate the individual words as if they are from another language. Just rearrange them into a natural English sentence.

Example:
ASL gloss: ME WANT GO STORE
English: I want to go to the store.

Rules:
- Output ONLY the final English sentence, nothing else
- Treat every input word as a literal English sign label
- Do NOT interpret words as Japanese, Spanish, or any other language
- No explanations, reasoning, or extra text

ASL gloss: ${glossInput}
English:`;

        const apiResponse = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: {
                    temperature: 0.2,
                    maxOutputTokens: 50,
                    topP: 0.8,
                    thinkingConfig: {
                        thinkingLevel: "MINIMAL"
                    }
                }
            })
        });

        const payload = await apiResponse.json();

        if (!apiResponse.ok) {
            console.error("API Error Response:", payload);
            return res.status(apiResponse.status).json({
                error: "Google API Error",
                details: payload
            });
        }

        // Skip "thought" parts — thinking models sometimes sneak those in
        const parts = payload?.candidates?.[0]?.content?.parts || [];
        const finalPart = parts.filter(p => !p.thought).pop();
        let rawText = finalPart?.text || "";

        // Strip markdown artifacts, bullets, numbering, wrapping quotes
        let sentence = rawText
            .replace(/^[*\-•\d.\)\s"]+/gm, "")
            .replace(/^["']|["']$/g, "")
            .trim();

        // If multi-line, just grab the last real line (the actual answer)
        const lines = sentence.split("\n").filter(l => l.trim() !== "");
        sentence = (lines.pop() || "").trim();

        // Make sure we always return a proper sentence
        if (sentence) {
            sentence =
                sentence.charAt(0).toUpperCase() +
                sentence.slice(1);

            if (!sentence.endsWith(".")) {
                sentence += ".";
            }
        } else {
            sentence = "Unable to generate translation.";
        }

        return res.status(200).json({ translation: sentence });

    } catch (err) {
        console.error("Server Crash:", err);
        return res.status(500).json({
            error: "Failed to connect to the AI model."
        });
    }
};