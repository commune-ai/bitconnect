// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.3;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "interfaces/utils/access/IAccessControl.sol";

struct CreatorState {
  uint256[] itemIds;
}

struct Module {
  string key;
  string[] meta_cid;
  address owner;
  }
