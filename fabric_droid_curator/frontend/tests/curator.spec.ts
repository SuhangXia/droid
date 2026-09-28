import { expect, test } from "@playwright/test";

test("three-stream review, one-cut editing, immediate branch persistence and preview", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Fabric-DROID")).toBeVisible();
  await expect(page.locator(".video-card")).toHaveCount(3);
  await expect(page.locator(".timeline")).toBeVisible();
  await expect(page.locator(".cut-marker")).toHaveCount(1);
  await expect(page.locator(".charts canvas")).toHaveCount(3);

  const keep = page.getByRole("button", { name: /^Keep this episode/ });
  await keep.click();
  await expect(keep).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText(/Saved v/)).toBeVisible({ timeout: 5000 });

  const timestampBefore = await page.locator(".timestamp").textContent();
  await page.getByRole("button", { name: "Frame ›" }).click();
  await expect(page.locator(".timestamp")).not.toHaveText(timestampBefore || "");

  const remove = page.getByRole("button", { name: /Pick and remove/ });
  await remove.click();
  await expect(remove).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText(/Saved v/)).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".segment-list button")).toHaveCount(2);

  const cut = page.locator(".cut-marker");
  const timeline = page.locator(".timeline");
  const stableBox = await cut.boundingBox();
  const timelineBox = await timeline.boundingBox();
  if (!stableBox || !timelineBox) throw new Error("timeline marker is not measurable");
  await page.mouse.move(stableBox.x + 3, stableBox.y + 3);
  await page.mouse.down();
  await page.mouse.move(Math.min(timelineBox.x + timelineBox.width - 10, stableBox.x + 12), stableBox.y + 3);
  await page.mouse.up();

  await expect(page.getByText(/Saved v/)).toBeVisible({ timeout: 5000 });
  await page.reload();
  await expect(page.locator(".video-card")).toHaveCount(3);
  await expect(page.getByRole("button", { name: /Pick and remove/ })).toHaveAttribute("aria-pressed", "true");

  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Play" })).toBeVisible();
});
