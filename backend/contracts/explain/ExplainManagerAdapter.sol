// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;
import {ExplainSchema} from "contracts/explain/Structs.sol";
import {IExplainManager} from "interfaces/explain/IExplainManager.sol";

contract ExplainManagerAdapter {

    IExplainManager public explain;

    function connectExplain(address explainAddress) public {
        explain = IExplainManager(explainAddress);
    }

    function getExplainer(string memory name) external view returns(ExplainSchema memory )
    {
    return explain.getExplainer(name);
    }
    function getExplainers() external view returns(ExplainSchema[] memory )
    {
    return explain.getExplainers();
    }
     
}
