// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;

contract Creator {

    string public name ;
    address public walletAddress;
    string public avatarURI;

    constructor (string memory _name) {
        walletAddress = msg.sender;
        name = _name;
    }

    function addUserAvator(string memory _avatarURI) public {
        avatarURI = _avatarURI;
    }
    
}