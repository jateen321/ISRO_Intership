import { expect, test } from '@playwright/test';

test('renders the truthful empty analysis state', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: /See what changed/ })).toBeVisible();
  await expect(page.getByText('Waiting for two images')).toBeVisible();
  await expect(page.getByRole('button', { name: /Run assessment/ })).toBeDisabled();
  await expect(page.getByText(/not been trained on real satellite imagery/)).toBeVisible();
  await expect(page.getByText(/no image is ever uploaded to a server/)).toBeVisible();
});

test('upload controls are keyboard-accessible and expose image inputs', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Waiting for two images')).toBeVisible();

  const beforeButton = page.getByRole('button', { name: /Drop before disaster image/ });
  const afterButton = page.getByRole('button', { name: /Drop after disaster image/ });
  await beforeButton.focus();
  await expect(beforeButton).toBeFocused();
  await afterButton.focus();
  await expect(afterButton).toBeFocused();
  await expect(page.locator('input[type="file"]')).toHaveCount(2);
  await expect(page.locator('input[type="file"]').nth(0)).toHaveAttribute('accept', /image\/png/);
  await expect(page.locator('input[type="file"]').nth(1)).toHaveAttribute('aria-label', /after disaster image/i);
  await expect(page.getByRole('button', { name: /Run assessment/ })).toBeDisabled();
});
