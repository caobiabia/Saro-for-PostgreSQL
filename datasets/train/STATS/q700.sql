select  count(*) from postHistory as ph,          posts as p,  		users as u,  		badges as b  where u.Id = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = b.UserId  AND p.PostTypeId=2  AND p.ViewCount>=0  AND p.AnswerCount<=4  AND u.Reputation>=1  AND u.Reputation<=3150  AND u.Views>=0;