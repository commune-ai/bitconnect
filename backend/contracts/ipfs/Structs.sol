pragma solidity ^0.8.7;

// you can reference an object by storing the relative path within an ipfs node


struct IPNSObject {
    string path; // path in the object pinned to IPNS ({root_key}.{child_key_1}.{child_key_2})
    string uri; // uri of the ipfs node
}
