# Required Imports
import numpy as np
import random
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd


def extractRequirementsFromSourceFile(filePath):
    """
    This method reads the input excel file provided as input on which we need to perform Strategic Release Planning
    :param filePath: Input excel file that contains the strategic release plan details and the actual requirements list
    :return: The master list of all the requirements and the strategic release plan summary list
    """
    try:
        requirementList = []
        requirementCapacityList = []
        requirementColumnName = "Feature Description"
        excelData = pd.read_excel(filePath, sheet_name=None, header=None)

        for sheet_name in reversed(excelData.keys()):
            sheetData = excelData[sheet_name]
            sentenceList = sheetData.iloc[:, 0].tolist()
            if requirementColumnName in sentenceList:
                sentenceList.remove(requirementColumnName)
            if sheet_name != "Summary":
                requirementList.extend(sentenceList)
            else:
                (requirementCapacityList.extend(sentenceList), requirementCapacityList.reverse())[1]

        return list(set(requirementList)), requirementCapacityList
    except Exception as e:
        print("Exception in extractRequirementsFromSourceFile", e)
        raise e


def performSentimentAnalysis(requirementList):
    """
    This method performs Sentiment Analysis using vader_lexicon Algorithm
    :param requirementList: List of Requirements on which sentiment analysis needs to be performed.
    :return:
    List of compound scores for each requirement from the input list.
    """
    try:
        sentimentAnalysisAlgo = "vader_lexicon"
        compoundScoreList = []
        nltk.download(sentimentAnalysisAlgo)
        sia = SentimentIntensityAnalyzer()

        for eachRequirement in requirementList:
            score = sia.polarity_scores(eachRequirement)["compound"]
            compoundScoreIntegration = {
                "description": eachRequirement,
                "sentiment": score
            }
            compoundScoreList.append(compoundScoreIntegration)

        compoundScoreDf = pd.DataFrame(compoundScoreList)
        print(compoundScoreDf)
        return compoundScoreList
    except Exception as e:
        print("Exception in performSentimentAnalysis", e)
        raise e


# Reward function
def rewardFunction(planBalances, planId, action, sentiment):
    """
    The reward function used to perform Reinforcement Learning.
    :param planBalances: List of dictionaries equivalent to the number of release plans given as input which contains
    initial values of {'positive': 0, 'negative': 0}
    :param planId: Each Strategic Plan Identifier.
    :param action: action=1 requirement gets picked, then we check for the sentiment. If sentiment is good then positive
    reward else negative reward.
    :param sentiment:
    :return:
    The return value is the actual computed reward (Iteratively computed balanced requirement reward).
    """
    try:
        if action == 1:  # If selecting the requirement
            balance = planBalances[planId]
            if sentiment > 0:
                return -10 if balance["positive"] - balance["negative"] > 0 else 10
            else:
                return -10 if balance["negative"] - balance["positive"] > 0 else 10
        return 0  # No reward or penalty for not selecting
    except Exception as e:
        print("Exception in rewardFunction", e)
        raise e


def triggerStrategicReleasePlanning(filePath):
    """
    Master Method that invokes the actual strategic release planning for various Requirements based on plans.
    :param filePath: The input excel file path is provided as this parameter which contains both requirements and their
    strategic release plan threshold summary.
    :return:
    """
    try:
        # Q-learning parameters
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1
        episodes = 1000
        nActions = 2  # Actions: 0 = do not select, 1 = select
        globally_selected_requirements = set()
        requirementList, requirementCapacityList = extractRequirementsFromSourceFile(filePath)
        # print("requirementList: ", requirementList)
        # print("requirementCapacityList: ", requirementCapacityList)
        nPlans = len(requirementCapacityList)
        print("nPlans: ", nPlans)
        polarityScores = performSentimentAnalysis(requirementList)
        # print("polarityScores: ", polarityScores)
        nRequirements = len(requirementList)
        Qtable = np.zeros((nPlans, nRequirements, nActions))
        planBalances = [{"positive": 0, "negative": 0} for _ in range(nPlans)]
        # Q-learning algorithm
        for episode in range(episodes):
            selectedForPlan = [set() for _ in range(nPlans)]
            for planId in range(nPlans):
                for _ in range(requirementCapacityList[planId]):
                    state = random.randint(0, nRequirements - 1)
                    while state in selectedForPlan[planId]:
                        state = random.randint(0, nRequirements - 1)
                    action = 1 if random.uniform(0, 1) < epsilon else np.argmax(Qtable[planId, state])
                    sentiment = polarityScores[state]["sentiment"]
                    reward = rewardFunction(planBalances, planId, action, sentiment)
                    Qtable[planId, state, action] = (1 - alpha) * Qtable[planId, state, action] + alpha * (
                            reward + gamma * np.max(Qtable[planId, (state + 1) % nRequirements]))
                    if action == 1:
                        selectedForPlan[planId].add(state)
                        if sentiment > 0:
                            planBalances[planId]["positive"] += 1
                        else:
                            planBalances[planId]["negative"] += 1
        selectedRequirementsPerPlan = [[] for _ in range(nPlans)]
        for planId in range(nPlans):
            availableReqs = set(range(nRequirements)) - globally_selected_requirements
            for _ in range(requirementCapacityList[planId]):
                bestReq = None
                bestValue = float('-inf')
                for reqId in availableReqs:
                    value = np.max(Qtable[planId, reqId])
                    if value > bestValue:
                        bestValue = value
                        bestReq = reqId
                if bestReq is not None:
                    selectedRequirementsPerPlan[planId].append(polarityScores[bestReq]["description"])
                    availableReqs.remove(bestReq)
                    globally_selected_requirements.add(bestReq)
        # Display the selected requirements for each release plan
        for planId, selectedReqs in enumerate(selectedRequirementsPerPlan):
            print(f"Release Plan {planId + 1}:")
            print("Selected requirement count:", len(selectedReqs))
            for req in selectedReqs:
                print(f"  - {req}")
            print("\n")
    except Exception as e:
        print("Exception in triggerStrategicReleasePlanning", e)
        raise e


if __name__ == "__main__":
    print("Output for ASN3-Run1-Dataset: ")
    triggerStrategicReleasePlanning(
        "/Users/saikrishnanaraharasetti/PycharmProjects/collegeStuff/ASN3-Run1-Dataset (1).xlsx")
    print("Output for ASN3-Run2-Dataset: ")
    triggerStrategicReleasePlanning(
        "/Users/saikrishnanaraharasetti/PycharmProjects/collegeStuff/ASN3-Run2-Dataset (1).xlsx")
    print("Output for ASN3-Run3-Dataset: ")
    triggerStrategicReleasePlanning(
        "/Users/saikrishnanaraharasetti/PycharmProjects/collegeStuff/ASN3-Run3-Dataset (1).xlsx")
