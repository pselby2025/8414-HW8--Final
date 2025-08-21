# Testing Guide

This project uses manual tests to validate key behaviors.

## ✅ Test Case 1: Benign URL

**Input:**
- Features consistent with a legitimate site (e.g., long domain, no IP address, no suspicious structure).

**Expected Output:**
- Prediction: BENIGN
- Threat Attribution tab not displayed

## 🔥 Test Case 2: Malicious – State-Sponsored

**Input:**
- SSLfinal_State = 1
- Prefix_Suffix = 1

**Expected Output:**
- Prediction: MALICIOUS
- Threat Actor: State-Sponsored

## 💸 Test Case 3: Malicious – Organized Crime

**Input:**
- Shortining_Service = 1
- having_IP_Address = 1

**Expected Output:**
- Prediction: MALICIOUS
- Threat Actor: Organized Cybercrime

## 🎯 Test Case 4: Malicious – Hacktivist

**Input:**
- has_political_keyword = 1

**Expected Output:**
- Prediction: MALICIOUS
- Threat Actor: Hacktivist
