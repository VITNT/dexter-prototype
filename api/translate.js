module.exports = async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method Not Allowed" });
    }

    try {
        const aslString = req.body.aslString;

        if (!aslString) {
            return res.status(400).json({ error: "Missing ASL string" });
        }

        const API_KEY = process.env.GEMINI_API_KEY;

        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key=${API_KEY}`;

        const prompt = `
Translate this ASL gloss into natural English.

Rules:
- Output ONE short sentence only
- No explanations
- No reasoning
- No steps
- No extra text

ASL: ${aslString}
`;

        const response = await fetch(url, {
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

        const data = await response.json();

        if (!response.ok) {
            console.error("API Error Response:", data);
            return res.status(response.status).json({
                error: "Google API Error",
                details: data
            });
        }

        // ✅ Safe extraction — skip "thought" parts from thinking models
        const parts = data?.candidates?.[0]?.content?.parts || [];
        const outputPart = parts.filter(p => !p.thought).pop();
        let rawText = outputPart?.text || "";

        // 🔧 Light cleanup
        let cleaned = rawText
            // Remove markdown formatting / bullets / numbering
            .replace(/^[*\-•\d.\)\s"]+/gm, "")
            // Remove wrapping quotes
            .replace(/^["']|["']$/g, "")
            .trim();

        // If multi-line, keep only the last meaningful line
        const lines = cleaned.split("\n").filter(l => l.trim() !== "");
        cleaned = (lines.pop() || "").trim();

        // 🔧 Final formatting (guarantee sentence)
        if (cleaned) {
            cleaned =
                cleaned.charAt(0).toUpperCase() +
                cleaned.slice(1);

            if (!cleaned.endsWith(".")) {
                cleaned += ".";
            }
        } else {
            cleaned = "Unable to generate translation.";
        }

        return res.status(200).json({ translation: cleaned });

    } catch (error) {
        console.error("Server Crash:", error);
        return res.status(500).json({
            error: "Failed to connect to the AI model."
        });
    }
};