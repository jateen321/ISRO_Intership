import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  metadataBase: new URL('http://localhost:3000'),
  title: 'GeoShield AI — Satellite Damage Assessment',
  description: 'A browser-private research prototype for comparing pre- and post-disaster satellite imagery.',
  openGraph: {
    title: 'GeoShield AI — Satellite Damage Assessment',
    description: 'Compare before and after satellite imagery to understand building damage.',
    type: 'website',
  },
  twitter: {
    card: 'summary',
    title: 'GeoShield AI — Satellite Damage Assessment',
    description: 'A browser-private research prototype for satellite damage assessment.',
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
