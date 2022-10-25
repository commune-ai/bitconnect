// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;

struct ModelURI {
    string explain; // explain object
    string model; // mdel object pointer (ie ipfs hash)
    string code;
}

struct ModelState {
    string name; // token id representing model
    ModelURI uri; // modeluri
}