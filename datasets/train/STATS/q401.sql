select  count(*) from comments as c,  		posts as p,           postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = p.LastEditorUserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND ph.PostHistoryTypeId=3  AND p.ViewCount=24  AND p.AnswerCount>=0  AND p.AnswerCount<=3;