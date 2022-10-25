// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.3;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "interfaces/utils/access/IAccessControl.sol";
import {CreatorState, CommuneItem} from "./Structs.sol";

contract ModelCommune is ReentrancyGuard {
  
  address payable owner;
  uint256 nextItemId = 0;
  string public name;
  string public category;
  mapping(address=>CreatorState) creatorStates;
  mapping(uint256 => CommuneItem) communeItems;
  mapping(uint256=>uint256) public id2index;
  mapping(address=>uint256) public model2id;
  uint256[] public itemIds;

  constructor(string memory _name) {
    owner = payable(msg.sender);
    name = _name;
  }

  function addItem(address modelContract) public payable nonReentrant {
    if (model2id[modelContract]==0) {
    nextItemId ++;
    uint256 itemId = nextItemId;

    // ensure msg.sender has owner role for ModelNFT contract
    // require(ModelNFT(nftContract).hasRole("owner", msg.sender),
    //          "msg.sender does not have owner priveledges");

    itemIds.push(nextItemId);
    id2index[itemId] = itemIds.length-1;
    creatorStates[msg.sender].itemIds.push(itemId);
    communeItems[itemId] =  CommuneItem(
      itemId,
      modelContract,
      msg.sender
      );
    model2id[modelContract] = itemId;  
    }
  }
  function removeItem(uint256 itemId) public payable nonReentrant {

    // ensure msg.sender has owner role for ModelNFT contract
    // require(ModelNFT(nftContract).hasRole("owner", msg.sender),
    //          "msg.sender does not have owner priveledges");

    if (model2id[communeItems[itemId].model]!=0) {
      uint256[] storage creatorTokenIds = creatorStates[msg.sender].itemIds;
      for (uint i; i<creatorTokenIds.length;i++) {
        if (creatorTokenIds[i] == itemId) {
          creatorTokenIds[i] = creatorTokenIds[creatorTokenIds.length-1];
          creatorTokenIds.pop();
        }
      }

      // swap last id with current id
      id2index[itemIds[itemIds.length-1]] = id2index[itemId];
      itemIds[id2index[itemId]] = itemIds[itemIds.length-1];
      delete id2index[itemId];
      itemIds.pop();

      delete model2id[communeItems[itemId].model];
      delete communeItems[itemId];
      
    
    }

  }
  
  /* Returns all unsold items items */
  function listItems() public view returns (CommuneItem[] memory) {
    uint itemCount = nextItemId;

    CommuneItem[] memory items = new CommuneItem[](itemCount);
    for (uint i = 0; i < itemIds.length; i++) {
        CommuneItem storage currentItem = communeItems[itemIds[i]];
        items[i] = currentItem;      
    }
    return items;
  }
  function listMyItems() public view returns (CommuneItem[] memory) {
    uint256[] storage creatorItemIds = creatorStates[msg.sender].itemIds;
    CommuneItem[] memory items = new CommuneItem[](creatorItemIds.length);
    for (uint i = 0; i < creatorItemIds.length; i++) {
        items[i] = communeItems[itemIds[i]];      
    }
    return items;
  }
  /* Returns onlyl items that a user has purchased */

  /* Returns only items a user has created */
}