// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.3;


struct CreatorState {
  uint256[] itemIds;
}

struct CommuneItem {
  uint256 id;
  address model;
  address creator;
  }