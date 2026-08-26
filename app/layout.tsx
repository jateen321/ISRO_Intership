import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('http://localhost:3000'),
  title: 'IncidentIQ — Application Intelligence',
  description: 'AI-powered incident detection, correlation, and root-cause analysis for modern engineering teams.',
  openGraph: {
    title: 'IncidentIQ — Application Intelligence',
    description: 'Detect incidents, correlate production signals, and explain root cause in seconds.',
    type: 'website',
    images: [{ url: '/og.png', width: 1732, height: 909, alt: 'IncidentIQ — AI-powered incident intelligence' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'IncidentIQ — Application Intelligence',
    description: 'Detect incidents, correlate production signals, and explain root cause in seconds.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
