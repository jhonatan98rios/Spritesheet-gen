import { ChatOpenAI } from "@langchain/openai";

export interface SpriteResult {
  csv: string;
  palette: Record<string, string>;
}

const SYSTEM_PROMPT = `You are a pixel art generator. You output ONLY valid JSON — no markdown, no explanation, no code fences. Your entire response must parse as JSON.

The JSON object has exactly two keys:

1. "csv" — a string containing exactly 64 rows, each row containing exactly 64 comma-separated integers from 0 to 9. Rows are joined by newline characters (\\n). No trailing newline. No spaces.
   - 0 means transparent/background/empty space. Use it generously for empty areas around the subject.
   - Values 1-9 represent different colors (defined in the palette).
   - The sprite must be a recognizable representation of the user's description, centered in the 64x64 grid.
   - Use solid blocks of color (no dithering, no gradients). Think retro 8-bit pixel art.
   - The subject should occupy roughly 30-60% of the grid area, centered.

2. "palette" — an object with keys "0" through "9" mapping to CSS color strings (hex like "#FF8844" or rgba like "rgba(255,136,68,1)").
   - Key "0" MUST be a fully transparent color: "rgba(0,0,0,0)" or "#00000000".
   - Keys "1" through "9" should be the colors used in the CSV, chosen to match the described subject.
   - Use distinct, vibrant colors so each index is clearly visible.
   - Every key "0" through "9" MUST be present.

Example output format:
{
  "csv": "0,0,0,1,1,0,0,0,...(64 values per row)\\n0,0,1,1,1,1,0,0,...\\n...(64 rows total)",
  "palette": {
    "0": "rgba(0,0,0,0)",
    "1": "#333333",
    "2": "#CC4444",
    "3": "#44CC44",
    "4": "#4444CC",
    "5": "#CCCC44",
    "6": "#CC44CC",
    "7": "#44CCCC",
    "8": "#FF8844",
    "9": "#FFFFFF"
  }
}

Rules:
- CSV must have EXACTLY 64 rows. Each row must have EXACTLY 64 values.
- Use ONLY the digits 0-9 in the CSV.
- All 10 palette keys ("0" through "9") MUST be present.
- Output ONLY the JSON object. No markdown wrapping, no explanation text.`;

function buildUserPrompt(description: string): string {
  return `Generate a 64x64 pixel art sprite of: ${description}`;
}

function parseResponse(text: string): SpriteResult {
  // Strip markdown code fences if the model ignores instructions
  let json = text.trim();
  if (json.startsWith("```")) {
    json = json.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }

  const parsed = JSON.parse(json);

  // Validate csv
  if (typeof parsed.csv !== "string") {
    throw new Error("Response missing 'csv' field");
  }
  const rows = parsed.csv.trim().split("\n");
  if (rows.length !== 64) {
    throw new Error(`CSV has ${rows.length} rows, expected 64`);
  }
  for (let i = 0; i < rows.length; i++) {
    const cols = rows[i].split(",");
    if (cols.length !== 64) {
      throw new Error(`CSV row ${i} has ${cols.length} columns, expected 64`);
    }
    for (const col of cols) {
      const n = parseInt(col, 10);
      if (isNaN(n) || n < 0 || n > 9) {
        throw new Error(`CSV row ${i} contains invalid value: "${col}"`);
      }
    }
  }

  // Validate palette
  if (typeof parsed.palette !== "object" || parsed.palette === null) {
    throw new Error("Response missing 'palette' field");
  }
  for (let i = 0; i <= 9; i++) {
    if (typeof parsed.palette[String(i)] !== "string") {
      throw new Error(`Palette missing key "${i}"`);
    }
  }

  return { csv: parsed.csv, palette: parsed.palette };
}

export async function generateSprite(
  description: string
): Promise<SpriteResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is not set");
  }

  const model = new ChatOpenAI({
    model: "deepseek-chat",
    temperature: 0.7,
    maxTokens: 16384,
    configuration: {
      baseURL: "https://api.deepseek.com/v1",
      apiKey,
    },
  });

  const response = await model.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: buildUserPrompt(description) },
  ]);

  const text =
    typeof response.content === "string"
      ? response.content
      : Array.isArray(response.content)
        ? response.content.map((c) => (typeof c === "string" ? c : JSON.stringify(c))).join("")
        : "";

  return parseResponse(text);
}
