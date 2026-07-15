"use client";

import { useEffect, useRef } from "react";
import { SPRITE_CSV } from "./sprite-data";

const PALETTE: Record<number, string> = {
  0: "rgba(0,0,0,0)",
  1: "#555555",
  2: "#8B4513",
  3: "#CC0000",
  4: "#FF8800",
  5: "#FFDD00",
  6: "#00AA00",
  7: "#3366CC",
  8: "#8833AA",
  9: "#FFFFFF",
};

const SCALE = 4;

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = SPRITE_CSV.trim().split("\n");
    for (let y = 0; y < rows.length; y++) {
      const cols = rows[y].split(",");
      for (let x = 0; x < cols.length; x++) {
        const colorIndex = parseInt(cols[x], 10);
        const color = PALETTE[colorIndex];
        if (!color || colorIndex === 0) continue;
        ctx.fillStyle = color;
        ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
      <canvas ref={canvasRef} width={256} height={256} className="border border-zinc-700" />
    </div>
  );
}
