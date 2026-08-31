import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import Home from './page';

describe('GeoShield analysis shell', () => {
  it('starts in an empty state without fabricated results', () => {
    render(<Home />);
    expect(screen.getByRole('heading', { name: /see what changed/i })).toBeInTheDocument();
    expect(screen.getByText(/waiting for two images/i)).toBeInTheDocument();
    expect(screen.getByText(/your assessment will appear here/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run assessment/i })).toBeDisabled();
    expect(screen.getAllByText('Undamaged').length).toBeGreaterThan(0);
    expect(screen.getByText(/placeholder model/i)).toBeInTheDocument();
  });

  it('shows a ready state after two image files are selected', () => {
    render(<Home />);
    const before = new File(['before'], 'before.png', { type: 'image/png' });
    const after = new File(['after'], 'after.png', { type: 'image/png' });
    fireEvent.change(screen.getByLabelText(/choose before disaster image/i), { target: { files: [before] } });
    fireEvent.change(screen.getByLabelText(/choose after disaster image/i), { target: { files: [after] } });
    expect(screen.getByText(/pair ready for assessment/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run assessment/i })).toBeEnabled();
    expect(screen.getByAltText(/before disaster uploaded preview/i)).toBeInTheDocument();
    expect(screen.getByAltText(/after disaster uploaded preview/i)).toBeInTheDocument();
  });
});
