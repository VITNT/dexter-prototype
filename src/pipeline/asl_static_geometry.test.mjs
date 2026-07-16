import assert from "node:assert/strict";
import {
    classifyClosedThumbLetter,
    hasCrossedIndexMiddle,
    measureClosedThumbPlacement
} from "./asl_static_geometry.mjs";

function point(x, y, z = 0) {
    return { x, y, z };
}

function transformHand(hand, { angle = 0, mirrored = false, scale = 0.18 } = {}) {
    const radians = angle * Math.PI / 180;
    const cosine = Math.cos(radians);
    const sine = Math.sin(radians);
    return hand.map(source => {
        const anatomicalX = mirrored ? -source.x : source.x;
        const rotatedX = anatomicalX * cosine - source.y * sine;
        const rotatedY = anatomicalX * sine + source.y * cosine;
        return point(
            0.5 + rotatedX * scale,
            0.55 - rotatedY * scale,
            source.z * scale
        );
    });
}

function closedHand(letter, noise = 0) {
    const hand = Array.from({ length: 21 }, () => point(0.5, -0.4, 0));
    hand[0] = point(0.5, -1.05, 0);

    const fingerXs = [0, 0.33, 0.66, 1];
    for (let finger = 0; finger < 4; finger++) {
        const base = 5 + finger * 4;
        const x = fingerXs[finger];
        hand[base] = point(x, 0, 0);
        hand[base + 1] = point(x + noise * (finger % 2 ? -1 : 1), 0.3, -0.01);
        hand[base + 2] = point(x, 0.19, -0.015);
        hand[base + 3] = point(x - noise, 0.06, 0);
    }

    const thumbTips = {
        A: point(-0.2, 0.36, 0),
        E: point(0.2, 0.07, -0.01),
        S: point(0.5, 0.24, -0.18),
        T: point(0.165, 0.23, 0.06),
        N: point(0.495, 0.23, 0.06),
        M: point(0.83, 0.23, 0.06)
    };
    hand[1] = point(-0.15, -0.18, 0);
    hand[2] = point(-0.1, 0, 0);
    hand[3] = point(0, 0.12, 0);
    hand[4] = thumbTips[letter];
    return hand;
}

for (const letter of ["A", "E", "S", "T", "N", "M"]) {
    for (const mirrored of [false, true]) {
        for (const scale of [0.1, 0.18, 0.3]) {
            for (const angle of [-42, 0, 37]) {
                const hand = transformHand(closedHand(letter, 0.008), { angle, mirrored, scale });
                assert.equal(
                    classifyClosedThumbLetter(hand),
                    letter,
                    `${letter} must survive ${mirrored ? "mirroring" : "normal view"} at ${angle} degrees and scale ${scale}.`
                );
            }
        }
    }
}

const ambiguousSlot = closedHand("T");
ambiguousSlot[4] = point(0.33, 0.23, 0.06);
assert.equal(
    classifyClosedThumbLetter(transformHand(ambiguousSlot)),
    null,
    "A thumb exactly between T and N must be rejected instead of flickering."
);

const ambiguousFront = closedHand("N");
ambiguousFront[4] = point(0.495, 0.23, -0.055);
assert.equal(
    classifyClosedThumbLetter(transformHand(ambiguousFront)),
    null,
    "An unclear tucked/on-top thumb must be rejected instead of overlapping S and N."
);

function twoFingerHand(crossed) {
    const hand = closedHand("N");
    hand[5] = point(0, 0, 0);
    hand[9] = point(0.33, 0, 0);
    if (crossed) {
        hand[6] = point(0.1, 0.32, 0);
        hand[10] = point(0.23, 0.32, 0);
        hand[7] = point(0.29, 0.63, -0.03);
        hand[11] = point(0.13, 0.63, 0.03);
        hand[8] = point(0.4, 0.94, -0.03);
        hand[12] = point(0.05, 0.94, 0.03);
    } else {
        hand[6] = point(0.05, 0.32, 0);
        hand[10] = point(0.28, 0.32, 0);
        hand[7] = point(0.06, 0.63, 0);
        hand[11] = point(0.27, 0.63, 0);
        hand[8] = point(0.07, 0.94, 0);
        hand[12] = point(0.26, 0.94, 0);
    }
    return hand;
}

for (const mirrored of [false, true]) {
    for (const scale of [0.1, 0.18, 0.3]) {
        for (const angle of [-55, -20, 0, 43, 70]) {
            assert.equal(
                hasCrossedIndexMiddle(transformHand(twoFingerHand(true), { angle, mirrored, scale })),
                true,
                `R must remain crossed at ${angle} degrees and scale ${scale}.`
            );
            assert.equal(
                hasCrossedIndexMiddle(transformHand(twoFingerHand(false), { angle, mirrored, scale })),
                false,
                `U must not become R at ${angle} degrees and scale ${scale}.`
            );
        }
    }
}

const sMeasurement = measureClosedThumbPlacement(transformHand(closedHand("S")));
const nMeasurement = measureClosedThumbPlacement(transformHand(closedHand("N")));
assert.ok(sMeasurement.thumbFrontness > 0.08);
assert.ok(nMeasurement.thumbFrontness < 0);

console.log("ASL static geometry tests passed: S/M/N/T slots and crossed R are rotation-safe.");
