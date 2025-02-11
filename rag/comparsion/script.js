document.addEventListener('DOMContentLoaded', function() {
    fetch('comparison_output.json') // Fetch the JSON file created by Python
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('comparison-sections').innerHTML = `<p class="error-message">Error loading comparison results. Check console for details and 'comparison_output.json' for raw response.</p><pre>${JSON.stringify(data, null, 4)}</pre>`;
                console.error("Error in comparison data:", data); // Log error to console
                return; // Exit if there's an error
            }

            document.getElementById('summary1-file').textContent = data.summary1_filename;
            document.getElementById('summary2-file').textContent = data.summary2_filename;

            const comparisonSectionsDiv = document.getElementById('comparison-sections');

            for (const categoryKey in data.comparison_result) {
                if (data.comparison_result.hasOwnProperty(categoryKey) && categoryKey !== 'OverallRecommendation') {
                    const categoryData = data.comparison_result[categoryKey];
                    const categoryDiv = document.createElement('div');
                    categoryDiv.classList.add('comparison-category');

                    const titleElement = document.createElement('h2');
                    titleElement.classList.add('category-title');
                    titleElement.textContent = categoryKey.replace(/([A-Z])/g, ' $1').trim(); // Add space before capital letters for display
                    categoryDiv.appendChild(titleElement);

                    if (categoryData.differences && categoryData.differences.length > 0) {
                        const differencesSection = createSection('Differences', categoryData.differences);
                        categoryDiv.appendChild(differencesSection);
                    }
                    if (categoryData.favorableAgreement) {
                        const favorableAgreementSection = createFavorableAgreementSection(categoryData.favorableAgreement);
                        categoryDiv.appendChild(favorableAgreementSection);
                    }
                    if (categoryData.concernPoints && categoryData.concernPoints.length > 0) {
                        const concernPointsSection = createConcernPointsSection('Concern Points', categoryData.concernPoints);
                        categoryDiv.appendChild(concernPointsSection);
                    }
                    if (categoryData.missingInformation && categoryData.missingInformation.length > 0) {
                        const missingInformationSection = createMissingInformationSection('Missing Information', categoryData.missingInformation);
                        categoryDiv.appendChild(missingInformationSection);
                    }
                     if (categoryData.recommendations && categoryData.recommendations.length > 0) {
                        const recommendationsSection = createRecommendationsSection('Recommendations', categoryData.recommendations);
                        categoryDiv.appendChild(recommendationsSection);
                    }


                    comparisonSectionsDiv.appendChild(categoryDiv);
                }
            }

            // Handle Overall Recommendation separately (outside the loop)
            if (data.comparison_result.OverallRecommendation) {
                const overallRecommendationData = data.comparison_result.OverallRecommendation;
                const overallDiv = document.createElement('div');
                overallDiv.classList.add('comparison-category');

                const titleElement = document.createElement('h2');
                titleElement.classList.add('category-title');
                titleElement.textContent = "Overall Recommendation";
                overallDiv.appendChild(titleElement);

                if (overallRecommendationData.summary) {
                    const summarySection = createSection('Summary', [overallRecommendationData.summary]); // Wrap summary in array for createSection function
                    overallDiv.appendChild(summarySection);
                }
                 if (overallRecommendationData.agreementRecommendation) {
                    const overallFavorableAgreementSection = createOverallFavorableAgreementSection(overallRecommendationData.agreementRecommendation);
                    overallDiv.appendChild(overallFavorableAgreementSection);
                }
                if (overallRecommendationData.keyTakeaways && overallRecommendationData.keyTakeaways.length > 0) {
                    const takeawaysSection = createRecommendationsSection('Key Takeaways', overallRecommendationData.keyTakeaways); // Reusing recommendations style
                    overallDiv.appendChild(takeawaysSection);
                }

                comparisonSectionsDiv.appendChild(overallDiv);
            }


        })
        .catch(error => {
            document.getElementById('comparison-sections').innerHTML = `<p class="error-message">Failed to load comparison data. Please ensure 'comparison_output.json' is in the same directory and accessible.</p><pre>${error}</pre>`;
            console.error("Fetch error:", error);
        });
});


function createSection(title, items) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = title;
    sectionDiv.appendChild(sectionTitle);
    const list = document.createElement('ul');
    items.forEach(item => {
        const listItem = document.createElement('li');
        listItem.classList.add('section-item');
        listItem.textContent = item.point ? `${item.point}: ${item.detail} Example: ${item.example || 'N/A'}` : item; // Handle both string and object formats in differences
        list.appendChild(listItem);
    });
    sectionDiv.appendChild(list);
    return sectionDiv;
}
function createFavorableAgreementSection(agreement) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = "Favorable Agreement";
    sectionDiv.appendChild(sectionTitle);
    const p = document.createElement('p');
    p.classList.add('section-item', 'favorable-agreement-highlight');
    p.textContent = agreement;
    sectionDiv.appendChild(p);
    return sectionDiv;
}
function createOverallFavorableAgreementSection(agreement) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = "Overall Agreement Recommendation";
    sectionDiv.appendChild(sectionTitle);
    const p = document.createElement('p');
    p.classList.add('section-item', 'favorable-agreement-highlight');
    p.textContent = agreement;
    sectionDiv.appendChild(p);
    return sectionDiv;
}


function createConcernPointsSection(title, items) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = title;
    sectionDiv.appendChild(sectionTitle);
    const list = document.createElement('ul');
    items.forEach(item => {
        const listItem = document.createElement('li');
        listItem.classList.add('section-item', 'concern-point-highlight');
        listItem.textContent = item;
        list.appendChild(listItem);
    });
    sectionDiv.appendChild(list);
    return sectionDiv;
}

function createMissingInformationSection(title, items) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = title;
    sectionDiv.appendChild(sectionTitle);
    const list = document.createElement('ul');
    items.forEach(item => {
        const listItem = document.createElement('li');
        listItem.classList.add('section-item', 'missing-info-highlight');
        listItem.textContent = item;
        list.appendChild(listItem);
    });
    sectionDiv.appendChild(list);
    return sectionDiv;
}
function createRecommendationsSection(title, items) {
    const sectionDiv = document.createElement('div');
    sectionDiv.classList.add('section');
    const sectionTitle = document.createElement('h3');
    sectionTitle.textContent = title;
    sectionDiv.appendChild(sectionTitle);
    const list = document.createElement('ul');
    items.forEach(item => {
        const listItem = document.createElement('li');
        listItem.classList.add('section-item', 'recommendation-highlight');
        listItem.textContent = item;
        list.appendChild(listItem);
    });
    sectionDiv.appendChild(list);
    return sectionDiv;
}