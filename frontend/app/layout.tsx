import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pedestrian Detection",
  description: "A Next.js app for pedestrian detection",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div>
          <div>
            {children}
          </div>
        </div>
      </body>

    </html>
  );
}
